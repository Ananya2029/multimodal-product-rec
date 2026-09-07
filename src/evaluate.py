"""
evaluate.py
Shared Recall@K / NDCG@K metrics for the sequential recommendation task
(used after adapting fused embeddings into SASRec/GRU4Rec via RecBole,
or for any ranked-list evaluation).
"""
import numpy as np


def recall_at_k(ranked_items, ground_truth, k=10):
    """ranked_items: list of ranked lists (one per query). ground_truth: list of single true items."""
    hits = [1 if gt in ranked[:k] else 0 for ranked, gt in zip(ranked_items, ground_truth)]
    return float(np.mean(hits))


def ndcg_at_k(ranked_items, ground_truth, k=10):
    scores = []
    for ranked, gt in zip(ranked_items, ground_truth):
        if gt in ranked[:k]:
            rank = ranked[:k].index(gt) + 1
            scores.append(1 / np.log2(rank + 1))
        else:
            scores.append(0.0)
    return float(np.mean(scores))


def run_full_eval(ranked_items, ground_truth, ks=(10, 20)):
    results = {}
    for k in ks:
        results[f"Recall@{k}"] = recall_at_k(ranked_items, ground_truth, k)
        results[f"NDCG@{k}"] = ndcg_at_k(ranked_items, ground_truth, k)
    return results


def modality_dropout_eval(fusion_module, z_img, z_txt, drop="image"):
    """
    Robustness test: zero out one modality at inference time and re-run fusion,
    to see which fusion strategy (concat vs gated) degrades more gracefully.
    """
    import torch
    z_img_d = torch.zeros_like(z_img) if drop == "image" else z_img
    z_txt_d = torch.zeros_like(z_txt) if drop == "text" else z_txt
    with torch.no_grad():
        z_fused, _ = fusion_module(z_img_d, z_txt_d)
    return z_fused


if __name__ == "__main__":
    # quick smoke test with dummy data
    ranked = [["a", "b", "c"], ["x", "y", "z"]]
    gts = ["b", "w"]
    print(run_full_eval(ranked, gts, ks=(3,)))
