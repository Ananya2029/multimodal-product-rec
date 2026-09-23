"""Model comparison.

Task A - item-to-item recommendation ("more like this"):
    every catalog product is a query; relevant = other products with the same articleType
    (strict: same articleType + baseColour). Metrics: P@5, P@10, mAP@10, NDCG@10.

Task B - text search (cross-modal):
    queries are shopper-style phrases built from metadata, e.g. "navy blue shirts for men";
    relevant = products matching colour + article type + gender.
    Text-only systems match against titles, CLIP/SigLIP image systems match against *images*
    (zero-shot), ResNet/DINOv2 cannot take a text query.

Also: fusion-weight sweep for the multimodal systems, and encoding speed.
"""
import json
import pickle

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import EMB_DIR, RESULTS_DIR
from .data import load_catalog
from .encoders import get_encoder
from .recommender import SYSTEMS, Recommender, System

K = 10
SHARED = ("CLIP", "SigLIP")


def ranking_metrics(rel: np.ndarray, n_rel: np.ndarray, k=K):
    """rel: (Q, k) binary relevance of ranked results; n_rel: total relevant items per query."""
    cum = np.cumsum(rel, 1)
    ap = ((cum / np.arange(1, k + 1)) * rel).sum(1) / np.minimum(n_rel, k).clip(min=1)
    disc = 1 / np.log2(np.arange(2, k + 2))
    idcg = np.array([disc[:min(int(n), k)].sum() for n in n_rel])
    ndcg = (rel * disc).sum(1) / np.where(idcg == 0, 1, idcg)
    return {"P@5": rel[:, :5].mean(), f"P@{k}": rel.mean(), f"mAP@{k}": ap.mean(), f"NDCG@{k}": ndcg.mean()}


def topk_from_scores(S, k=K, exclude_self=False):
    S = S.copy()
    if exclude_self:
        np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    order = np.take_along_axis(S, idx, 1).argsort(1)[:, ::-1]
    return np.take_along_axis(idx, order, 1)


def eval_item2item(rec: Recommender, labels: np.ndarray):
    top = topk_from_scores(rec.item_scores(), exclude_self=True)
    rel = (labels[top] == labels[:, None]).astype(float)
    n_rel = pd.Series(labels).map(pd.Series(labels).value_counts()).values - 1
    return ranking_metrics(rel, n_rel)


def build_text_queries(df: pd.DataFrame, min_items=3):
    keys = (df["baseColour"].astype(str) + "|" + df["articleType"] + "|" + df["gender"].astype(str)).values
    counts = pd.Series(keys).value_counts()
    who = {"Men": "for men", "Women": "for women", "Boys": "for boys", "Girls": "for girls", "Unisex": "unisex"}
    queries = []
    for key in counts[counts >= min_items].index:
        colour, art, gender = key.split("|")
        queries.append((f"{colour} {art} {who.get(gender, '')}".lower().strip(), key))
    return queries, keys


def eval_text_search(system: System, rec: Recommender, queries, item_keys, text_cache):
    q_txt = text_cache.get(system.text_model) if system.text_model else None
    q_txt_img = text_cache.get(system.image_model) if system.image_model in SHARED else None
    if q_txt is None and q_txt_img is None:
        return None  # image-only model with no text tower
    S = np.stack([rec.query_scores(q_txt=None if q_txt is None else q_txt[i],
                                   q_txt_imgenc=None if q_txt_img is None else q_txt_img[i])
                  for i in range(len(queries))])
    top = topk_from_scores(S)
    qkeys = np.array([k for _, k in queries])
    rel = (item_keys[top] == qkeys[:, None]).astype(float)
    n_rel = pd.Series(item_keys).value_counts().reindex(qkeys).values
    return ranking_metrics(rel, n_rel)


def main():
    df = load_catalog()
    art = df["articleType"].values
    art_col = (df["articleType"] + "|" + df["baseColour"].astype(str)).values
    queries, item_keys = build_text_queries(df)
    print(f"{len(df)} products | {df['articleType'].nunique()} article types | {len(queries)} text queries")

    qtexts = [q for q, _ in queries]
    text_cache = {name: get_encoder(name).encode_text(qtexts) for name in ("MiniLM", "CLIP", "SigLIP")}
    with open(EMB_DIR / "tfidf.pkl", "rb") as f:
        text_cache["TF-IDF"] = pickle.load(f).encode_text(qtexts)

    rows = []
    for s in SYSTEMS:
        rec = Recommender(s, 0.5)
        r = {"system": s.name, "group": s.group}
        r |= {f"i2i {k}": v for k, v in eval_item2item(rec, art).items()}
        r |= {f"i2i-strict {k}": v for k, v in eval_item2item(rec, art_col).items()}
        ts = eval_text_search(s, rec, queries, item_keys, text_cache)
        r |= {f"search {k}": v for k, v in (ts or {}).items()}
        rows.append(r)
        print(f"  {s.name:30s} i2i={r['i2i NDCG@10']:.3f}  strict={r['i2i-strict NDCG@10']:.3f}"
              + (f"  search={r['search NDCG@10']:.3f}" if ts else "  search=n/a"))
    res = pd.DataFrame(rows)

    timing_file = RESULTS_DIR / "encode_timing.json"
    if timing_file.exists():
        timing = json.loads(timing_file.read_text())
        res["encode ms/item"] = [
            sum(timing.get(f"{m}_{mod}", {}).get("ms_per_item", 0)
                for m, mod in ((s.image_model, "image"), (s.text_model, "text")) if m)
            for s in SYSTEMS]
    res.round(4).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    # fusion weight sweep for multimodal systems
    sweep = []
    for s in [s for s in SYSTEMS if s.group == "multimodal"]:
        for w in np.round(np.linspace(0, 1, 11), 2):
            rec = Recommender(s, w)
            sweep.append({"system": s.name, "image_weight": w,
                          "i2i NDCG@10": eval_item2item(rec, art)["NDCG@10"],
                          "i2i-strict NDCG@10": eval_item2item(rec, art_col)["NDCG@10"]})
    sweep = pd.DataFrame(sweep)
    sweep.round(4).to_csv(RESULTS_DIR / "fusion_sweep.csv", index=False)

    plot(res, sweep)
    print("\nBest image weight per multimodal system (strict):")
    print(sweep.loc[sweep.groupby("system")["i2i-strict NDCG@10"].idxmax()].round(4).to_string(index=False))
    print(f"\nSaved results to {RESULTS_DIR}")


def plot(res, sweep):
    colors = {"text": "#4C72B0", "image": "#DD8452", "multimodal": "#55A868"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    panels = [("i2i NDCG@10", "Item-to-item: same article type"),
              ("i2i-strict NDCG@10", "Item-to-item: same type + colour"),
              ("search NDCG@10", "Text search: colour + type + gender")]
    d = res.iloc[::-1].reset_index(drop=True)
    for ax, (col, title) in zip(axes, panels):
        raw = d[col] if col in d else pd.Series([np.nan] * len(d))
        ax.barh(d["system"], raw.fillna(0), color=[colors[g] for g in d["group"]])
        for y, v in enumerate(raw):
            ax.text((0 if pd.isna(v) else v) + 0.01, y, "n/a" if pd.isna(v) else f"{v:.3f}", va="center", fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("NDCG@10")
        ax.set_xlim(0, 1.1)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()], colors.keys(),
               loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col in zip(axes, ["i2i NDCG@10", "i2i-strict NDCG@10"]):
        for name, g in sweep.groupby("system"):
            ax.plot(g["image_weight"], g[col], marker="o", label=name)
        ax.set_xlabel("image weight (0 = text only, 1 = image only)")
        ax.set_ylabel(col)
        ax.set_title(f"Fusion weight sweep: {col}")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fusion_sweep.png", dpi=130)


if __name__ == "__main__":
    main()
