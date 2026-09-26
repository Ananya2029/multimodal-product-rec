"""Ranking metrics, text-query construction and significance tests for product recommendation."""
import numpy as np
import pandas as pd

K = 10


def ranking_metrics(rel: np.ndarray, n_rel: np.ndarray, k=K):
    """rel: (Q, k) binary relevance of ranked results; n_rel: total relevant items per query."""
    cum = np.cumsum(rel, 1)
    ap = ((cum / np.arange(1, k + 1)) * rel).sum(1) / np.minimum(n_rel, k).clip(min=1)
    disc = 1 / np.log2(np.arange(2, k + 2))
    idcg = np.array([disc[:min(int(n), k)].sum() for n in n_rel])
    ndcg = (rel * disc).sum(1) / np.where(idcg == 0, 1, idcg)
    return {"P@5": rel[:, :5].mean(), f"P@{k}": rel.mean(), f"mAP@{k}": ap.mean(), f"NDCG@{k}": ndcg.mean(),
            "_ndcg": ndcg}  # per-query NDCG, used for confidence intervals / significance tests


def public(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


def bootstrap_ci(x: np.ndarray, n_boot=1000, seed=0):
    """95% bootstrap confidence interval of the mean."""
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_p(a: np.ndarray, b: np.ndarray, n_boot=2000, seed=0):
    """Two-sided p-value that mean(a) != mean(b), resampling the *same* queries for both systems."""
    rng = np.random.default_rng(seed)
    d = a - b
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))


def topk_from_scores(S, k=K, exclude_self=False):
    S = S.copy()
    if exclude_self:
        np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    order = np.take_along_axis(S, idx, 1).argsort(1)[:, ::-1]
    return np.take_along_axis(idx, order, 1)


def build_text_queries(df: pd.DataFrame, min_items=3):
    keys = (df["baseColour"].astype(str) + "|" + df["articleType"] + "|" + df["gender"].astype(str)).values
    counts = pd.Series(keys).value_counts()
    who = {"Men": "for men", "Women": "for women", "Boys": "for boys", "Girls": "for girls", "Unisex": "unisex"}
    queries = []
    for key in counts[counts >= min_items].index:
        colour, art, gender = key.split("|")
        queries.append((f"{colour} {art} {who.get(gender, '')}".lower().strip(), key))
    return queries, keys
