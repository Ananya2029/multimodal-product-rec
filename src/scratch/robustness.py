"""Noisy / missing title benchmark: how each trained model copes when catalog titles are damaged.

Real catalogs have short, missing or wrong titles. We corrupt the titles of the (unseen) TEST products and
re-score every model. Photos are untouched, so the image branch is what can compensate.

Conditions (each averaged over 3 random draws):
    clean            original titles
    drop 30/60%      each title word removed with that probability
    no title         every title removed (drop 100%)
    wrong 20/50%     that share of products gets the title of another, random product

Metrics (NDCG@10): similar products by type and by type + colour (catalog items are the queries), and a
clean shopper text query against the damaged catalog.

    python -m src.scratch.robustness
Works from the exported checkpoints and data/scratch/test_images (no training data needed); CPU is fine.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..evaluate import build_text_queries, ranking_metrics, topk_from_scores
from . import serve
from .data import SCRATCH_DIR, encode, square, tokenize
from .models import METHOD_NAMES, MultimodalRec
from .train import to_input

CONDITIONS = [("clean", "drop", 0.0), ("drop 30%", "drop", 0.3), ("drop 60%", "drop", 0.6),
              ("no title", "drop", 1.0), ("wrong 20%", "wrong", 0.2), ("wrong 50%", "wrong", 0.5)]
DRAWS = 3


def corrupt(titles: list[str], kind: str, rate: float, rng: np.random.Generator) -> list[str]:
    if rate == 0:
        return list(titles)
    if kind == "drop":
        return [" ".join(w for w in tokenize(t) if rng.random() >= rate) for t in titles]
    out = list(titles)  # "wrong": swap in the title of a random other product
    idx = np.flatnonzero(rng.random(len(titles)) < rate)
    for i in idx:
        j = int(rng.integers(len(titles)))
        out[i] = titles[j if j != i else (j + 1) % len(titles)]
    return out


def ndcg(S: np.ndarray, q_labels, item_labels, n_rel, exclude_self) -> tuple[float, np.ndarray]:
    top = topk_from_scores(S, exclude_self=exclude_self)
    rel = (item_labels[top] == np.asarray(q_labels)[:, None]).astype(float)
    m = ranking_metrics(rel, n_rel)
    return float(m["NDCG@10"]), m["_ndcg"]


@torch.no_grad()
def run(models: dict[str, str], out_csv, batch=512, seed=0):
    """models: {display key: checkpoint file name in models/scratch}."""
    meta = pd.read_csv(serve.EMB_DIR / "test_meta.csv").reset_index(drop=True)
    vocab = json.loads((SCRATCH_DIR / "vocab.json").read_text())
    imgs = np.stack([square(Image.open(SCRATCH_DIR / "test_images" / f"{i}.jpg")) for i in meta["id"]])
    titles = meta["productDisplayName"].astype(str).tolist()
    y_t = pd.factorize(meta["articleType"])[0]
    y_f = pd.factorize(meta["articleType"] + "|" + meta["baseColour"].astype(str))[0]
    n_t = pd.Series(y_t).map(pd.Series(y_t).value_counts()).values - 1
    n_f = pd.Series(y_f).map(pd.Series(y_f).value_counts()).values - 1
    queries, item_keys = build_text_queries(meta, min_items=2)
    qkeys = np.array([k for _, k in queries])
    q_nrel = pd.Series(item_keys).value_counts().reindex(qkeys).values
    q_tok = torch.from_numpy(encode([q for q, _ in queries], vocab)).long()

    # corrupted title sets: clean once, every other condition DRAWS times
    rng = np.random.default_rng(seed)
    variants = [(name, d, encode(corrupt(titles, kind, rate, rng), vocab))
                for name, kind, rate in CONDITIONS for d in range(1 if rate == 0 else DRAWS)]

    rows = []
    for key, ckpt in models.items():
        method = key.split("+")[0]
        m = MultimodalRec(method, len(vocab))
        m.load_state_dict(torch.load(serve.MODEL_DIR / ckpt, map_location="cpu"))
        m.eval()
        F = [np.zeros((len(meta), 256 if method == "late" else 128), np.float32) for _ in variants]
        for b in range(0, len(meta), batch):
            sl = slice(b, b + batch)
            x = to_input(torch.from_numpy(imgs[sl]), False) if m.use_img else None
            img_out = m.img(x) if m.use_img else None     # photos encoded once per batch
            for v, (_, _, tok) in enumerate(variants):
                t = torch.from_numpy(tok[sl]).long() if m.use_txt else None
                F[v][sl] = m.encode(None, t, img_out=img_out)["f"].numpy()
        Q = m.encode(None, q_tok)["f"].numpy() if m.use_txt else None
        for v, (name, draw, _) in enumerate(variants):
            S = F[v] @ F[v].T
            r = {"model": key, "condition": name, "draw": draw,
                 "i2i": ndcg(S, y_t, y_t, n_t, True)[0], "i2i_strict": ndcg(S, y_f, y_f, n_f, True)[0],
                 "text2item": ndcg(Q @ F[v].T, qkeys, item_keys, q_nrel, False)[0] if Q is not None else np.nan}
            rows.append(r)
        print(f"  {key:10s} done", flush=True)
    df = pd.DataFrame(rows)
    df.round(5).to_csv(out_csv, index=False)
    return df


def summarize(df: pd.DataFrame, res_dir, names=None):
    names = names or {}
    agg = df.groupby(["model", "condition"])[["i2i", "i2i_strict", "text2item"]].agg(["mean", "std"])
    agg.columns = [f"{a} {b}" for a, b in agg.columns]
    agg = agg.reset_index()
    order = [c for c, _, _ in CONDITIONS]
    agg["condition"] = pd.Categorical(agg["condition"], order, ordered=True)
    agg = agg.sort_values(["model", "condition"])
    agg.round(4).to_csv(res_dir / "robustness_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, (conds, title) in zip(axes, [(order[:4], "Title words removed"),
                                         (["clean", "wrong 20%", "wrong 50%"], "Wrong titles")]):
        for model, g in agg.groupby("model", observed=True):
            g = g[g["condition"].isin(conds)]
            lw, style = (3, "-") if model.startswith("gated") else (1.6, "--" if model in ("image", "text") else "-")
            ax.errorbar(range(len(g)), g["i2i mean"], yerr=g["i2i std"].fillna(0), label=names.get(model, model),
                        lw=lw, ls=style, marker="o", capsize=3)
        ax.set_xticks(range(len(conds)), conds)
        ax.set_title(f"{title}: similar products (same type)")
        ax.set_ylabel("NDCG@10")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(res_dir / "robustness.png", dpi=140)
    plt.close(fig)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="extra key=checkpoint.pt pairs (e.g. balanced models)")
    a = ap.parse_args()
    exp = serve.export_info()["methods"]
    models = {k: v["checkpoint"] for k, v in exp.items()}
    models.update(dict(x.split("=", 1) for x in a.extra))
    df = run(models, serve.RES_DIR / "robustness_runs.csv")
    agg = summarize(df, serve.RES_DIR, {**METHOD_NAMES})
    piv = agg.pivot(index="model", columns="condition", values="i2i mean")
    print("\nSimilar products (same type), NDCG@10 mean over draws:")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
