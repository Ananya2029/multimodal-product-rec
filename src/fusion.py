"""Trained multimodal fusion head, compared across three optimizers (SGD, Adam, AdamW).

The pretrained CLIP encoders stay frozen. A small MLP learns to map [CLIP image ; CLIP text] (1024-d)
to a 256-d joint embedding where products of the same type (and colour) are close, using a
supervised contrastive loss. Every optimizer gets the same data, model, epochs, seeds and cosine schedule;
only its own hyper-parameters (learning rate, weight decay) are tuned, on validation data.

Protocol (fair comparison):
  1. products are split 70 / 30 (train / test); the test products are never seen during training or tuning
  2. each optimizer gets its own small hyper-parameter search on a validation split carved out of train
  3. the tuned optimizers are retrained on the full train split with 3 seeds; test metrics are mean +- std
The zero-shot baselines are scored on the same test products.

    python -m src.fusion
"""
from __future__ import annotations

import json
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import embeddings
from .config import EMB_DIR, RESULTS_DIR, ROOT, SEED
from .data import load_catalog

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

OPTIMIZERS = ["SGD", "Adam", "AdamW"]
SEARCH_SPACE = {  # hyper-parameters tried per optimizer (picked on validation NDCG)
    "SGD": [{"lr": lr} for lr in (0.01, 0.05, 0.2)],
    "Adam": [{"lr": lr} for lr in (3e-4, 1e-3, 3e-3)],
    "AdamW": [{"lr": lr, "weight_decay": wd} for lr in (3e-4, 1e-3, 3e-3) for wd in (0.05, 0.5)],
}
SEEDS = [0, 1, 2]
EPOCHS = 40
BATCH = 256
TEMPERATURE = 0.1
MODALITY_DROPOUT = 0.3  # teach the head to cope with text-only / image-only queries


class FusionHead(nn.Module):
    def __init__(self, d_in=1024, d_hidden=1024, d_out=256, p_drop=0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(p_drop),
                                 nn.Linear(d_hidden, d_out))
        self.skip = nn.Linear(d_in, d_out, bias=False)  # residual linear path keeps CLIP's geometry

    def forward(self, x):
        return F.normalize(self.skip(x) + self.mlp(x), dim=-1)


def supcon_loss(z: torch.Tensor, y: torch.Tensor, t=TEMPERATURE) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., 2020): same label = positive pair."""
    sim = z @ z.T / t
    self_mask = torch.eye(len(z), dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)
    pos = (y[:, None] == y[None, :]) & ~self_mask
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    has_pos = pos.any(1)
    loss = -(log_prob * pos).sum(1)[has_pos] / pos.sum(1)[has_pos]
    return loss.mean()


def make_optimizer(name: str, params, lr: float, weight_decay: float = 0.0):
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr)  # no weight decay: plain Adam
    if name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)  # decoupled weight decay
    raise ValueError(name)


def modality_dropout(img, txt, p=MODALITY_DROPOUT):
    """With prob p, replace one modality by the other (CLIP puts both in one space)."""
    r = torch.rand(len(img))
    img = torch.where((r < p / 2)[:, None], txt, img)
    txt = torch.where(((r >= p / 2) & (r < p))[:, None], img, txt)
    return torch.cat([img, txt], 1)


# ------------------------------------------------------------------ data
def load_split():
    df = load_catalog()
    I, T = embeddings.load("CLIP", "image"), embeddings.load("CLIP", "text")
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=df["articleType"])
    return df, np.asarray(I), np.asarray(T), np.sort(tr), np.sort(te)


def encode_labels(values):
    return torch.tensor(pd.factorize(values)[0])


# ------------------------------------------------------------------ evaluation on the held-out split
def ndcg_i2i(Z: np.ndarray, labels: np.ndarray, k=10) -> float:
    from .evaluate import eval_item2item

    class _R:  # minimal adapter for eval_item2item
        def item_scores(self):
            return Z @ Z.T
    return float(eval_item2item(_R(), labels)["NDCG@10"])


def test_metrics(Z, df_te) -> dict:
    art = df_te["articleType"].values
    art_col = (df_te["articleType"] + "|" + df_te["baseColour"].astype(str)).values
    return {"i2i NDCG@10": ndcg_i2i(Z, art), "i2i-strict NDCG@10": ndcg_i2i(Z, art_col)}


def text_search_metric(q_vecs: np.ndarray, Z: np.ndarray, qkeys, item_keys) -> float:
    from .evaluate import ranking_metrics, topk_from_scores
    top = topk_from_scores(q_vecs @ Z.T)
    rel = (item_keys[top] == np.array(qkeys)[:, None]).astype(float)
    n_rel = pd.Series(item_keys).value_counts().reindex(qkeys).values
    return float(ranking_metrics(rel, n_rel)["NDCG@10"])


# ------------------------------------------------------------------ training
def train_one(opt_name, hp, X_tr, y_type, y_fine, X_te, df_te, epochs=EPOCHS, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FusionHead()
    opt = make_optimizer(opt_name, model.parameters(), **hp)
    steps_per_epoch = int(np.ceil(len(X_tr) / BATCH))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    X_tr_t, X_te_t = torch.tensor(X_tr), torch.tensor(X_te)
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        losses = []
        for b in range(steps_per_epoch):
            j = perm[b * BATCH:(b + 1) * BATCH]
            x = modality_dropout(X_tr_t[j, :512], X_tr_t[j, 512:])
            z = model(x)
            loss = supcon_loss(z, y_type[j]) + 0.5 * supcon_loss(z, y_fine[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            Z = model(X_te_t).numpy()
        history.append({"optimizer": opt_name, "seed": seed, "epoch": epoch, "train_loss": float(np.mean(losses)),
                        **test_metrics(Z, df_te)})
    return model, history, time.perf_counter() - t0


@torch.no_grad()
def project(model, img: np.ndarray | None, txt: np.ndarray | None) -> np.ndarray:
    """Joint embedding for a query; a missing modality is replaced by the other (shared CLIP space)."""
    img = txt if img is None else img
    txt = img if txt is None else txt
    x = torch.tensor(np.hstack([np.atleast_2d(img), np.atleast_2d(txt)]), dtype=torch.float32)
    return model.eval()(x).numpy()


def load_head(opt_name: str) -> FusionHead:
    m = FusionHead()
    m.load_state_dict(torch.load(MODEL_DIR / f"fusion_{opt_name}.pt", map_location="cpu"))
    return m.eval()


def main():
    from .encoders import CLIPEncoder
    from .evaluate import build_text_queries

    df, I, T, tr, te = load_split()
    X = np.hstack([I, T]).astype(np.float32)
    df_te = df.iloc[te].reset_index(drop=True)
    y_type = encode_labels(df.iloc[tr]["articleType"].values)
    y_fine = encode_labels((df.iloc[tr]["articleType"] + "|" + df.iloc[tr]["baseColour"].astype(str)).values)
    print(f"train {len(tr)} products | test {len(te)} products (unseen during training)")

    # text-search queries restricted to the test split
    queries, item_keys = build_text_queries(df_te, min_items=2)
    qtexts, qkeys = [q for q, _ in queries], [k for _, k in queries]
    clip = CLIPEncoder()
    Q = clip.encode_text(qtexts)

    rows, histories = [], []

    # zero-shot baselines on the same test split
    zs = (0.5 ** 0.5) * np.hstack([I[te], T[te]])  # cos = 0.5 cos_img + 0.5 cos_txt
    zs_q = (0.5 ** 0.5) * np.hstack([Q, Q])
    rows.append({"model": "CLIP image + text (zero-shot, no training)", "optimizer": "-",
                 **test_metrics(zs, df_te), "search NDCG@10": text_search_metric(zs_q, zs, qkeys, item_keys),
                 "train time (s)": 0.0})
    D, M = embeddings.load("DINOv2", "image"), embeddings.load("MiniLM", "text")
    dm = (0.5 ** 0.5) * np.hstack([D[te], M[te]])
    rows.append({"model": "DINOv2 + MiniLM (zero-shot, no training)", "optimizer": "-",
                 **test_metrics(dm, df_te), "search NDCG@10": np.nan, "train time (s)": 0.0})

    # 1) hyper-parameter search per optimizer on a validation split of the TRAIN products
    tr_fit, val = train_test_split(tr, test_size=0.2, random_state=SEED, stratify=df.iloc[tr]["articleType"])
    df_val = df.iloc[val].reset_index(drop=True)
    fit_type = encode_labels(df.iloc[tr_fit]["articleType"].values)
    fit_fine = encode_labels((df.iloc[tr_fit]["articleType"] + "|" + df.iloc[tr_fit]["baseColour"].astype(str)).values)
    tuning, best_hp = [], {}
    print("hyper-parameter search (validation split):")
    for name in OPTIMIZERS:
        for hp in SEARCH_SPACE[name]:
            _, hist, _ = train_one(name, hp, X[tr_fit], fit_type, fit_fine, X[val], df_val, seed=SEED)
            score = hist[-1]["i2i NDCG@10"] + hist[-1]["i2i-strict NDCG@10"]
            tuning.append({"optimizer": name, **hp, "val i2i NDCG@10": hist[-1]["i2i NDCG@10"],
                           "val i2i-strict NDCG@10": hist[-1]["i2i-strict NDCG@10"], "val score": score})
            print(f"  {name:6s} {hp}  val score={score:.4f}")
        t = pd.DataFrame([r for r in tuning if r["optimizer"] == name])
        best = t.loc[t["val score"].idxmax()]
        best_hp[name] = {k: float(best[k]) for k in SEARCH_SPACE[name][0]}
    pd.DataFrame(tuning).round(5).to_csv(RESULTS_DIR / "optimizer_tuning.csv", index=False)
    print("best:", best_hp)

    # 2) retrain each tuned optimizer on the full train split with several seeds; score on test
    for name in OPTIMIZERS:
        runs = []
        for seed in SEEDS:
            model, hist, secs = train_one(name, best_hp[name], X[tr], y_type, y_fine, X[te], df_te, seed=seed)
            histories += hist
            Z_te = project(model, I[te], T[te])
            runs.append({**test_metrics(Z_te, df_te),
                         "search NDCG@10": text_search_metric(project(model, None, Q), Z_te, qkeys, item_keys),
                         "train time (s)": secs})
            if seed == SEEDS[0]:  # keep the first seed's model for the app
                torch.save(model.state_dict(), MODEL_DIR / f"fusion_{name}.pt")
                np.save(EMB_DIR / f"Fusion-{name}_joint.npy", project(model, I, T).astype(np.float32))
        runs = pd.DataFrame(runs)
        row = {"model": f"Trained fusion ({name})", "optimizer": name,
               "hyper-parameters": ", ".join(f"{k}={v:g}" for k, v in best_hp[name].items())}
        for c in runs.columns:
            row[c] = runs[c].mean()
            if c != "train time (s)":
                row[f"{c} std"] = runs[c].std()
        rows.append(row)
        print(f"  {name:6s} i2i={row['i2i NDCG@10']:.4f}+-{row['i2i NDCG@10 std']:.4f}  "
              f"strict={row['i2i-strict NDCG@10']:.4f}+-{row['i2i-strict NDCG@10 std']:.4f}  "
              f"search={row['search NDCG@10']:.4f}+-{row['search NDCG@10 std']:.4f}")

    res = pd.DataFrame(rows)
    res.round(4).to_csv(RESULTS_DIR / "optimizer_comparison.csv", index=False)
    pd.DataFrame(histories).round(4).to_csv(RESULTS_DIR / "optimizer_history.csv", index=False)
    (RESULTS_DIR / "fusion_split.json").write_text(json.dumps({"train": tr.tolist(), "test": te.tolist()}))
    plot(pd.DataFrame(histories), res)
    print("\n" + res.round(3).to_string(index=False))


def plot(h: pd.DataFrame, res: pd.DataFrame):
    colors = {"SGD": "#4C72B0", "Adam": "#DD8452", "AdamW": "#55A868"}
    h = h[h["seed"] == SEEDS[0]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, col, title in zip(axes, ["train_loss", "i2i NDCG@10", "i2i-strict NDCG@10"],
                              ["Training loss", "Test NDCG@10: same article type",
                               "Test NDCG@10: same type + colour"]):
        for name, g in h.groupby("optimizer"):
            ax.plot(g["epoch"], g[col], label=name, color=colors[name], lw=2)
        if col != "train_loss":
            base = res.iloc[0][col]
            ax.axhline(base, ls="--", color="grey", lw=1.2, label="CLIP zero-shot (no training)")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("supervised contrastive loss")
    axes[1].legend()
    fig.suptitle("Optimizer comparison: same model, data, epochs and schedule; each optimizer tuned on validation",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "optimizer_curves.png", dpi=130)


if __name__ == "__main__":
    main()
