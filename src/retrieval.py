"""
retrieval.py
Builds a FAISS index over fused (or single-modality) item embeddings and
evaluates item-to-item retrieval quality against ground-truth co-purchase /
co-session pairs derived from the interaction sequences.

Usage:
    python src/retrieval.py --config configs/default.yaml --embeddings fused_gated_embeddings.npy
"""
import argparse
import os
import json
import yaml
import numpy as np
import faiss
from collections import defaultdict


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_index(mat):
    mat = mat.astype("float32")
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    return index


def build_ground_truth_pairs(processed_dir, window=3):
    """Items co-occurring within `window` steps of a user's session = 'also relevant' ground truth."""
    with open(os.path.join(processed_dir, "sequences_test.json")) as f:
        sequences = json.load(f)
    gt = defaultdict(set)
    for seq in sequences.values():
        for i in range(len(seq)):
            for j in range(max(0, i - window), min(len(seq), i + window + 1)):
                if i != j:
                    gt[seq[i]].add(seq[j])
    return gt


def precision_at_k(retrieved, relevant, k):
    if not relevant:
        return None
    hits = len(set(retrieved[:k]) & relevant)
    return hits / k


def ndcg_at_k(retrieved, relevant, k):
    if not relevant:
        return None
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(processed_dir, embeddings_file, top_k=(1, 5, 10)):
    mat = np.load(os.path.join(processed_dir, embeddings_file))
    with open(os.path.join(processed_dir, embeddings_file.replace(".npy", "_asins.txt"))) as f:
        asins = f.read().splitlines()
    asin_to_idx = {a: i for i, a in enumerate(asins)}

    index = build_index(mat.copy())
    gt = build_ground_truth_pairs(processed_dir)

    results = {k: {"precision": [], "ndcg": []} for k in top_k}
    max_k = max(top_k) + 1  # +1 to drop self-match

    query_vecs = mat.astype("float32").copy()
    faiss.normalize_L2(query_vecs)
    _, I = index.search(query_vecs, max_k)

    for qi, asin in enumerate(asins):
        if asin not in gt:
            continue
        retrieved = [asins[j] for j in I[qi] if asins[j] != asin][:max(top_k)]
        relevant = gt[asin]
        for k in top_k:
            p = precision_at_k(retrieved, relevant, k)
            n = ndcg_at_k(retrieved, relevant, k)
            if p is not None:
                results[k]["precision"].append(p)
                results[k]["ndcg"].append(n)

    print(f"\nRetrieval results for: {embeddings_file}")
    for k in top_k:
        p_mean = np.mean(results[k]["precision"]) if results[k]["precision"] else float("nan")
        n_mean = np.mean(results[k]["ndcg"]) if results[k]["ndcg"] else float("nan")
        print(f"  Precision@{k}: {p_mean:.4f} | NDCG@{k}: {n_mean:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--embeddings", default="fused_gated_embeddings.npy",
                         help="filename inside data/processed/, e.g. image_embeddings.npy, "
                              "text_embeddings.npy, fused_concat_embeddings.npy, fused_gated_embeddings.npy")
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate_retrieval(cfg["data"]["processed_dir"], args.embeddings, tuple(cfg["retrieval"]["top_k"]))
