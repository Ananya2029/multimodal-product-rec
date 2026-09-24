"""Conditional DCGAN that generates new 64x64 product images for a chosen category.

Use in the project: "design a new product" -> the generator creates a concept image for a category,
and the recommender retrieves the most similar real products from the catalog.

Architecture
  Generator:      z (100) + category embedding -> 4 transposed-conv blocks -> 64x64x3 (tanh)
  Discriminator:  4 strided-conv blocks with spectral norm, minibatch-std feature,
                  projection conditioning (Miyato & Koyama, 2018)
  Loss:           hinge loss
  Optimizer:      Adam with TTUR (Heusel et al., 2017): D lr=4e-4, G lr=1e-4, betas=(0.0, 0.9)
  Small-data fix: DiffAugment (Zhao et al., 2020) on real AND fake images, so the discriminator can't
                  memorise the 2.7k training photos (without it the generator mode-collapsed)

Evaluation (in CLIP embedding space)
  FD-CLIP:        Frechet distance between CLIP features of real vs generated images (lower = more realistic)
  Class accuracy: a classifier trained on REAL images predicts the category of GENERATED images
                  (higher = the generator really draws the requested category)

    python -m src.gan                 # ~30 min on an 8-core CPU (35 s/epoch)
    python -m src.gan --epochs 200    # longer training; much faster on a GPU, e.g. Google Colab
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils import spectral_norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import IMAGE_DIR, RESULTS_DIR, ROOT, SEED
from .data import load_catalog

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
PROGRESS_DIR = RESULTS_DIR / "gan_progress"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    import os
    torch.set_num_threads(os.cpu_count() or 4)
Z_DIM, IMG = 100, 64
MIN_PER_CLASS = 40


# ------------------------------------------------------------------ data
def square_pad(im: Image.Image, size=IMG) -> Image.Image:
    """Catalog photos are portrait with white backgrounds: pad to a square with white, then resize."""
    im = im.convert("RGB")
    s = max(im.size)
    canvas = Image.new("RGB", (s, s), (255, 255, 255))
    canvas.paste(im, ((s - im.width) // 2, (s - im.height) // 2))
    return canvas.resize((size, size), Image.BICUBIC)


def load_gan_data():
    df = load_catalog()
    counts = df["subCategory"].value_counts()
    classes = sorted(counts[counts >= MIN_PER_CLASS].index)
    df = df[df["subCategory"].isin(classes)].reset_index(drop=True)
    x = np.stack([np.asarray(square_pad(Image.open(IMAGE_DIR / p))) for p in df["image_path"]])
    x = torch.tensor(x).permute(0, 3, 1, 2).float() / 127.5 - 1  # [-1, 1]
    y = torch.tensor(df["subCategory"].map({c: i for i, c in enumerate(classes)}).values)
    return x, y, classes


def to_pil(x: torch.Tensor) -> list[Image.Image]:
    x = ((x.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(a) for a in x]


# ------------------------------------------------------------------ DiffAugment (Zhao et al., 2020)
def diff_augment(x: torch.Tensor) -> torch.Tensor:
    """Differentiable colour + translation + cutout augmentation, applied to real and fake images alike."""
    n = x.size(0)
    dev = x.device
    # colour: brightness, saturation, contrast
    x = x + (torch.rand(n, 1, 1, 1, device=dev) - 0.5)
    mean = x.mean(1, keepdim=True)
    x = (x - mean) * (torch.rand(n, 1, 1, 1, device=dev) * 2) + mean
    mean = x.mean([1, 2, 3], keepdim=True)
    x = (x - mean) * (torch.rand(n, 1, 1, 1, device=dev) + 0.5) + mean
    # translation by up to 1/8 of the image, padding with white (+1) like the photo background
    shift = IMG // 8
    tx = torch.randint(-shift, shift + 1, (n,), device=dev)
    ty = torch.randint(-shift, shift + 1, (n,), device=dev)
    padded = F.pad(x, [shift] * 4, value=1.0)
    idx = torch.arange(IMG, device=dev)
    rows = (idx[None, :] + shift - ty[:, None]).clamp(0, IMG + 2 * shift - 1)
    cols = (idx[None, :] + shift - tx[:, None]).clamp(0, IMG + 2 * shift - 1)
    x = padded[torch.arange(n, device=dev)[:, None, None], :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
    # cutout of half the image size
    c = IMG // 2
    cy = torch.randint(0, IMG, (n, 1, 1), device=dev)
    cx = torch.randint(0, IMG, (n, 1, 1), device=dev)
    yy, xx = torch.meshgrid(idx, idx, indexing="ij")
    mask = ((yy[None] - cy).abs() >= c // 2) | ((xx[None] - cx).abs() >= c // 2)
    return x * mask[:, None].float()


@torch.no_grad()
def diversity(G, n_classes, n=8, seed=99) -> float:
    """Mean pairwise L1 distance between samples of the same class (0 = mode collapse)."""
    d = []
    for c in range(0, n_classes, max(1, n_classes // 4)):
        g = torch.Generator().manual_seed(seed + c)
        x = G.eval()(torch.randn(n, Z_DIM, generator=g), torch.full((n,), c, dtype=torch.long))
        d.append(torch.cdist(x.flatten(1), x.flatten(1), p=1).sum() / (n * (n - 1)) / x[0].numel())
    return float(torch.stack(d).mean())


# ------------------------------------------------------------------ models
class Generator(nn.Module):
    def __init__(self, n_classes, ngf=32, emb=50):
        super().__init__()
        self.embed = nn.Embedding(n_classes, emb)
        self.fc = nn.Linear(Z_DIM + emb, ngf * 8 * 4 * 4)
        self.bn0 = nn.Sequential(nn.BatchNorm2d(ngf * 8), nn.ReLU(True))
        self.ngf = ngf

        def up(i, o):
            return [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]
        self.net = nn.Sequential(*up(ngf * 8, ngf * 4), *up(ngf * 4, ngf * 2), *up(ngf * 2, ngf),
                                 nn.ConvTranspose2d(ngf, 3, 4, 2, 1), nn.Tanh())

    def forward(self, z, y):
        h = self.fc(torch.cat([z, self.embed(y)], 1)).view(-1, self.ngf * 8, 4, 4)
        return self.net(self.bn0(h))


class Discriminator(nn.Module):
    def __init__(self, n_classes, ndf=32):
        super().__init__()

        def down(i, o):
            return [spectral_norm(nn.Conv2d(i, o, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        self.net = nn.Sequential(*down(3, ndf), *down(ndf, ndf * 2), *down(ndf * 2, ndf * 4),
                                 *down(ndf * 4, ndf * 8))  # -> ndf*8 x 4 x 4
        d = ndf * 8 * 4 * 4 + 1  # +1: minibatch-std feature
        self.out = spectral_norm(nn.Linear(d, 1))
        self.embed = spectral_norm(nn.Embedding(n_classes, d))  # projection discriminator

    def forward(self, x, y):
        h = self.net(x).flatten(1)
        # minibatch standard deviation (Karras et al., 2018): lets D spot batches with too little variety
        mbstd = h.std(0, unbiased=False).mean().expand(len(h), 1)
        h = torch.cat([h, mbstd], 1)
        return self.out(h).squeeze(1) + (self.embed(y) * h).sum(1)


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) and hasattr(m, "weight"):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


# ------------------------------------------------------------------ generation API (used by the app / REST API)
def load_generator():
    meta = json.loads((MODEL_DIR / "gan_meta.json").read_text())
    G = Generator(len(meta["classes"]))
    G.load_state_dict(torch.load(MODEL_DIR / "gan_generator.pt", map_location="cpu"))
    return G.eval(), meta["classes"]


@torch.no_grad()
def generate(G, class_idx: int, n: int = 8, seed: int = 0) -> list[Image.Image]:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, Z_DIM, generator=g)
    y = torch.full((n,), class_idx, dtype=torch.long)
    return to_pil(G.eval()(z, y))


# ------------------------------------------------------------------ evaluation
def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


@torch.no_grad()
def evaluate(G, x_real, y_real, classes, n_per_class=30):
    """FD-CLIP and conditional class accuracy, with a real-vs-real reference and an untrained-G baseline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from .encoders import CLIPEncoder

    clip = CLIPEncoder()
    # up to 60 real images per category keeps CLIP encoding affordable on a CPU
    rng = np.random.default_rng(SEED)
    keep = np.concatenate([rng.permutation(np.where(y_real.numpy() == c)[0])[:60] for c in range(len(classes))])
    real_feats = clip.encode_image(to_pil(x_real[keep]))
    y_np = y_real.numpy()[keep]
    a, b = train_test_split(np.arange(len(y_np)), test_size=0.5, random_state=SEED, stratify=y_np)
    clf = LogisticRegression(max_iter=2000).fit(real_feats[a], y_np[a])

    def score(generator):
        ys = np.repeat(np.arange(len(classes)), n_per_class)
        imgs = []
        for c in range(len(classes)):
            imgs += generate(generator, c, n_per_class, seed=123 + c)
        f = clip.encode_image(imgs)
        return frechet_distance(real_feats, f), float((clf.predict(f) == ys).mean())

    fd, acc = score(G)
    untrained = Generator(len(classes))
    untrained.apply(weights_init)
    fd0, acc0 = score(untrained)
    return {
        "FD-CLIP (generated vs real)": fd,
        "FD-CLIP untrained generator (baseline)": fd0,
        "FD-CLIP real vs real (best possible)": frechet_distance(real_feats[a], real_feats[b]),
        "class accuracy (generated)": acc,
        "class accuracy untrained generator": acc0,
        "class accuracy real held-out images": float(clf.score(real_feats[b], y_np[b])),
        "chance accuracy": 1 / len(classes),
    }


# ------------------------------------------------------------------ training
def sample_grid(G, classes, path, n_per_class=6, seed=7, title=None):
    rows = len(classes)
    fig, axes = plt.subplots(rows, n_per_class, figsize=(n_per_class * 1.1, rows * 1.15))
    for c in range(rows):
        for k, im in enumerate(generate(G, c, n_per_class, seed=seed + c)):
            ax = axes[c, k]
            ax.imshow(im)
            ax.axis("off")
        axes[c, 0].text(-8, 32, classes[c], ha="right", va="center", fontsize=8)
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def real_diversity(x, y, n_classes, n=8) -> float:
    d = []
    for c in range(0, n_classes, max(1, n_classes // 4)):
        xs = x[y == c][:n].flatten(1)
        d.append(torch.cdist(xs, xs, p=1).sum() / (n * (n - 1)) / xs.shape[1])
    return float(torch.stack(d).mean())


def train(epochs: int = 50, batch: int = 64):
    torch.manual_seed(SEED)
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    x, y, classes = load_gan_data()
    print(f"{len(x)} images | {len(classes)} categories: {classes} | device={DEVICE}")
    print(f"real-data diversity (reference): {real_diversity(x, y, len(classes)):.3f}", flush=True)
    G, D = Generator(len(classes)).to(DEVICE), Discriminator(len(classes)).to(DEVICE)
    G.apply(weights_init)
    optG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))  # TTUR: slower generator
    optD = torch.optim.Adam(D.parameters(), lr=4e-4, betas=(0.0, 0.9))
    losses, t0 = [], time.perf_counter()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(x))
        for b in range(0, len(x) - batch + 1, batch):
            j = perm[b:b + batch]
            real, lab = x[j].to(DEVICE), y[j].to(DEVICE)
            real = torch.where(torch.rand(len(real), 1, 1, 1, device=DEVICE) < 0.5, real, real.flip(3))  # h-flip
            z = torch.randn(len(real), Z_DIM, device=DEVICE)
            fake = G(z, lab)
            # discriminator: hinge loss
            d_loss = (F.relu(1 - D(diff_augment(real), lab)).mean()
                      + F.relu(1 + D(diff_augment(fake.detach()), lab)).mean())
            optD.zero_grad()
            d_loss.backward()
            optD.step()
            # generator
            g_loss = -D(diff_augment(fake), lab).mean()
            optG.zero_grad()
            g_loss.backward()
            optG.step()
            losses.append({"epoch": epoch, "d_loss": d_loss.item(), "g_loss": g_loss.item()})
        el = time.perf_counter() - t0
        last = pd.DataFrame(losses[-10:]).mean()
        div = diversity(G.cpu(), len(classes))
        G.to(DEVICE).train()
        losses[-1]["diversity"] = div
        print(f"epoch {epoch:3d}/{epochs}  D={last['d_loss']:.3f}  G={last['g_loss']:.3f}  diversity={div:.3f}  "
              f"({el / epoch:.0f}s/epoch, ~{el / epoch * (epochs - epoch) / 60:.0f} min left)", flush=True)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            G.eval()
            sample_grid(G.cpu(), classes, PROGRESS_DIR / f"epoch_{epoch:03d}.png", title=f"epoch {epoch}")
            G.to(DEVICE).train()

    G = G.cpu().eval()
    torch.save(G.state_dict(), MODEL_DIR / "gan_generator.pt")
    (MODEL_DIR / "gan_meta.json").write_text(json.dumps({"classes": classes, "epochs": epochs}, indent=2))
    sample_grid(G, classes, RESULTS_DIR / "gan_samples.png", title=f"Conditional DCGAN samples ({epochs} epochs)")

    h = pd.DataFrame(losses).groupby("epoch").mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(h.index, h["d_loss"], label="discriminator (hinge)")
    ax.plot(h.index, h["g_loss"], label="generator")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("GAN training losses (Adam + TTUR: D lr 4e-4, G lr 1e-4)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "gan_losses.png", dpi=130)

    print("evaluating ...")
    metrics = evaluate(G, x, y, classes) | {"epochs": epochs, "train_minutes": round((time.perf_counter() - t0) / 60, 1),
                                            "n_images": len(x), "n_classes": len(classes)}
    (RESULTS_DIR / "gan_metrics.json").write_text(json.dumps(metrics, indent=2))
    for k, v in metrics.items():
        print(f"  {k:42s} {v:.4f}" if isinstance(v, float) else f"  {k:42s} {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    train(a.epochs, a.batch)
