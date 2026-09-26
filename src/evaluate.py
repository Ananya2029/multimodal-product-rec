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
from .metrics import (K, bootstrap_ci, build_text_queries, paired_bootstrap_p,  # noqa: F401
                      public, ranking_metrics, topk_from_scores)

SHARED = ("CLIP", "SigLIP")


def eval_item2item(rec: Recommender, labels: np.ndarray):
    top = topk_from_scores(rec.item_scores(), exclude_self=True)
    rel = (labels[top] == labels[:, None]).astype(float)
    n_rel = pd.Series(labels).map(pd.Series(labels).value_counts()).values - 1
    return ranking_metrics(rel, n_rel)


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
    per_query = {t: {} for t in TASKS}  # task -> system -> per-query NDCG@10
    for s in SYSTEMS:
        rec = Recommender(s, 0.5)
        r = {"system": s.name, "group": s.group}
        results = {"i2i": eval_item2item(rec, art), "i2i-strict": eval_item2item(rec, art_col),
                   "search": eval_text_search(s, rec, queries, item_keys, text_cache)}
        for task, m in results.items():
            if m is None:
                continue
            per_query[task][s.name] = m["_ndcg"]
            r |= {f"{task} {k}": v for k, v in public(m).items()}
            r[f"{task} NDCG@10 CI low"], r[f"{task} NDCG@10 CI high"] = bootstrap_ci(m["_ndcg"])
        ts = results["search"]
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
    summary = summarize(res, per_query)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # fusion weight sweep for multimodal systems
    sweep = []
    for s in [s for s in SYSTEMS if s.group == "multimodal"]:
        for w in np.round(np.linspace(0, 1, 11), 2):
            rec = Recommender(s, w)
            sweep.append({"system": s.name, "image_weight": w,
                          "i2i NDCG@10": float(eval_item2item(rec, art)["NDCG@10"]),
                          "i2i-strict NDCG@10": float(eval_item2item(rec, art_col)["NDCG@10"])})
    sweep = pd.DataFrame(sweep)
    sweep.round(4).to_csv(RESULTS_DIR / "fusion_sweep.csv", index=False)

    plot(res, sweep)
    print("\nBest image weight per multimodal system (strict):")
    print(sweep.loc[sweep.groupby("system")["i2i-strict NDCG@10"].idxmax()].round(4).to_string(index=False))
    print_summary(summary)
    print(f"\nSaved results to {RESULTS_DIR}")


TASKS = {
    "i2i": "More like this (same article type)",
    "i2i-strict": "More like this (same type + colour)",
    "search": "Text search (colour + type + gender)",
}


def summarize(res: pd.DataFrame, per_query: dict) -> dict:
    """Winner per task (with CI and a significance test against the runner-up) and an overall ranking."""
    out = {"tasks": {}, "overall": []}
    ranks = pd.DataFrame(index=res["system"])
    for task, label in TASKS.items():
        col = f"{task} NDCG@10"
        ranked = res.dropna(subset=[col]).sort_values(col, ascending=False)
        best, second = ranked.iloc[0], ranked.iloc[1]
        p = paired_bootstrap_p(per_query[task][best["system"]], per_query[task][second["system"]])
        out["tasks"][task] = {
            "label": label,
            "best": best["system"], "best_ndcg": round(float(best[col]), 4),
            "best_ci": [round(float(best[f"{col} CI low"]), 4), round(float(best[f"{col} CI high"]), 4)],
            "runner_up": second["system"], "runner_up_ndcg": round(float(second[col]), 4),
            "p_value": round(p, 4), "significant": bool(p < 0.05),
            "n_queries": int(len(per_query[task][best["system"]])),
            "ranking": [[n, round(float(v), 4)] for n, v in ranked[["system", col]].values],
        }
        # systems that can't do the task rank last
        ranks[task] = res.set_index("system")[col].rank(ascending=False, na_option="bottom")
    ranks["mean_rank"] = ranks.mean(1)
    for name, row in ranks.sort_values("mean_rank").iterrows():
        out["overall"].append({"system": name, "mean_rank": round(float(row["mean_rank"]), 2),
                               **{f"rank_{t}": int(row[t]) for t in TASKS}})
    out["recommended"] = out["overall"][0]["system"]
    return out


def print_summary(summary: dict):
    print("\n=== Which model is best? ===")
    for t in summary["tasks"].values():
        sig = "significant" if t["significant"] else "NOT significant"
        print(f"  {t['label']:38s} best: {t['best']} ({t['best_ndcg']:.3f}, 95% CI {t['best_ci'][0]:.3f}-"
              f"{t['best_ci'][1]:.3f}) vs {t['runner_up']} ({t['runner_up_ndcg']:.3f}), p={t['p_value']:.3f} {sig}")
    print("\n  Overall ranking (mean rank over the 3 tasks):")
    for i, o in enumerate(summary["overall"], 1):
        print(f"   {i:2d}. {o['system']:32s} {o['mean_rank']:.2f}")
    print(f"\n  Recommended system: {summary['recommended']}")


def plot(res, sweep):
    colors = {"text": "#4C72B0", "image": "#DD8452", "multimodal": "#55A868"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    panels = [("i2i NDCG@10", "Item-to-item: same article type"),
              ("i2i-strict NDCG@10", "Item-to-item: same type + colour"),
              ("search NDCG@10", "Text search: colour + type + gender")]
    d = res.iloc[::-1].reset_index(drop=True)
    for ax, (col, title) in zip(axes, panels):
        raw = d[col] if col in d else pd.Series([np.nan] * len(d))
        lo, hi = d.get(f"{col} CI low"), d.get(f"{col} CI high")
        xerr = None if lo is None else np.vstack([(raw - lo).fillna(0), (hi - raw).fillna(0)])
        ax.barh(d["system"], raw.fillna(0), color=[colors[g] for g in d["group"]],
                xerr=xerr, error_kw={"ecolor": "#333", "capsize": 3, "lw": 1})
        best = raw.idxmax()
        for y, v in enumerate(raw):
            label = "n/a" if pd.isna(v) else f"{v:.3f}" + ("  ★ best" if y == best else "")
            x = 0 if pd.isna(v) else (hi.iloc[y] if hi is not None else v)
            ax.text(x + 0.01, y, label, va="center", fontsize=9, fontweight="bold" if y == best else None)
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
