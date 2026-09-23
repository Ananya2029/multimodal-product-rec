"""Compute and cache catalog embeddings for every encoder."""
import argparse
import json
import pickle
import time
from functools import lru_cache

import numpy as np

from .config import EMB_DIR, IMAGE_DIR, RESULTS_DIR
from .data import load_catalog, product_text
from .encoders import ENCODERS, get_encoder


def emb_path(model: str, modality: str):
    return EMB_DIR / f"{model}_{modality}.npy"


def build(models=None, bench: int | None = None):
    """bench=N: only time the encoders on N items (no embeddings saved)."""
    df = load_catalog()
    texts = [product_text(r) for _, r in df.iterrows()]
    images = [str(IMAGE_DIR / p) for p in df["image_path"]]
    if bench:
        texts, images = texts[:bench], images[:bench]
    timing_file = RESULTS_DIR / "encode_timing.json"
    timing = json.loads(timing_file.read_text()) if timing_file.exists() else {}

    for name in models or ENCODERS:
        print(f"== {name}")
        enc = get_encoder(name)
        if name == "TF-IDF":
            enc.fit(texts)
            if not bench:
                with open(EMB_DIR / "tfidf.pkl", "wb") as f:
                    pickle.dump(enc, f)
        for modality, items, fn in (("text", texts, enc.encode_text), ("image", images, enc.encode_image)):
            if modality not in enc.modalities:
                continue
            t0 = time.perf_counter()
            X = fn(items)
            ms = 1000 * (time.perf_counter() - t0) / len(items)
            if not bench:
                np.save(emb_path(name, modality), X)
                load.cache_clear()
            timing[f"{name}_{modality}"] = {"ms_per_item": round(ms, 2), "dim": int(X.shape[1])}
            print(f"   {modality}: {X.shape}  {ms:.1f} ms/item")
            timing_file.write_text(json.dumps(timing, indent=2))  # save as we go
        del enc


@lru_cache(maxsize=None)
def load(model: str, modality: str):
    """Load cached embeddings once per process and share them (read-only) between recommenders."""
    p = emb_path(model, modality)
    if not p.exists():
        return None
    X = np.load(p)
    X.flags.writeable = False
    return X


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help=f"subset of {list(ENCODERS)}")
    ap.add_argument("--bench", type=int, default=None, help="only time encoders on N items")
    a = ap.parse_args()
    build(a.models, a.bench)
