"""Dataset for from-scratch training.

    python -m src.scratch.data --source hf      # full 44k catalog from Hugging Face (run on Colab)
    python -m src.scratch.data --source local   # the 3k local catalog (quick local tests)
    python -m src.scratch.data --source folder --path /kaggle/input/fashion-product-images-small
                                                # same 44k catalog from files (no internet needed)

Output (data/scratch/):
    images.npy      uint8 (N, S, S, 3) square-padded product photos (S = 64 by default)
    tokens.npy      int32 (N, MAX_LEN) title token ids (0 = padding, 1 = unknown word)
    meta.csv        id, title, labels, split (train / val / test)
    vocab.json      tokenizer vocabulary, built from TRAIN titles only
    test_images/    original-resolution JPEGs of the test products (for the app)
"""
from __future__ import annotations

import argparse
import io
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from ..config import CATALOG_CSV, DATA_DIR, DATASET_NAME, IMAGE_DIR, SEED

SCRATCH_DIR = DATA_DIR / "scratch"
IMG_SIZE = 64
MAX_LEN = 16
MIN_PER_CLASS = 20   # article types with fewer products are dropped (too few to split and evaluate)
MIN_WORD_FREQ = 2
PAD, UNK = 0, 1
META_COLS = ["id", "gender", "masterCategory", "subCategory", "articleType", "baseColour", "season",
             "usage", "productDisplayName"]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def build_vocab(titles, min_freq=MIN_WORD_FREQ) -> dict[str, int]:
    counts = Counter(w for t in titles for w in tokenize(t))
    words = sorted(w for w, c in counts.items() if c >= min_freq)
    return {"<pad>": PAD, "<unk>": UNK, **{w: i + 2 for i, w in enumerate(words)}}


def encode(texts, vocab, max_len=MAX_LEN) -> np.ndarray:
    out = np.zeros((len(texts), max_len), dtype=np.int32)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, UNK) for w in tokenize(t)][:max_len]
        out[i, :len(ids)] = ids
    return out


def square(im: Image.Image, size=IMG_SIZE) -> np.ndarray:
    """Pad to a white square (photos have white backgrounds), then resize."""
    im = im.convert("RGB")
    s = max(im.size)
    canvas = Image.new("RGB", (s, s), (255, 255, 255))
    canvas.paste(im, ((s - im.width) // 2, (s - im.height) // 2))
    return np.asarray(canvas.resize((size, size), Image.BICUBIC), dtype=np.uint8)


def _rows_hf(limit=None):
    from datasets import load_dataset
    ds = load_dataset(DATASET_NAME, split="train")
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        ex = ds[i]
        if ex.get("image") is None or not ex.get("productDisplayName"):
            continue
        yield {c: ex.get(c) for c in META_COLS}, ex["image"]


def _rows_folder(root, limit=None):
    """The same catalog as files: styles.csv + images/<id>.jpg (e.g. the Kaggle dataset
    'paramaggarwal/fashion-product-images-small', usable without internet access)."""
    from pathlib import Path
    root = Path(root)
    styles = next(root.rglob("styles.csv"))
    img_dir = next(p for p in root.rglob("images") if p.is_dir())
    df = pd.read_csv(styles, on_bad_lines="skip")  # a few rows of the original CSV are malformed
    df = df.dropna(subset=["productDisplayName"])
    n = 0
    for _, r in df.iterrows():
        f = img_dir / f"{int(r['id'])}.jpg"
        if not f.exists():
            continue
        yield {c: r.get(c) for c in META_COLS}, Image.open(f)
        n += 1
        if limit and n >= limit:
            break


def _rows_local(limit=None):
    df = pd.read_csv(CATALOG_CSV)
    for _, r in df.head(limit).iterrows() if limit else df.iterrows():
        yield {c: r[c] for c in META_COLS}, Image.open(IMAGE_DIR / r["image_path"])


def prepare(source="hf", img_size=IMG_SIZE, limit=None, min_per_class=MIN_PER_CLASS, out=SCRATCH_DIR, path=None):
    from tqdm import tqdm
    out.mkdir(parents=True, exist_ok=True)
    rows, images, originals = [], [], {}
    rows_iter = {"hf": lambda: _rows_hf(limit), "local": lambda: _rows_local(limit),
                 "folder": lambda: _rows_folder(path, limit)}[source]()
    for meta, im in tqdm(rows_iter, desc="Reading products"):
        rows.append(meta)
        images.append(square(im, img_size))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=90)  # keep compressed bytes, not decoded images
        originals[meta["id"]] = buf.getvalue()
    df = pd.DataFrame(rows)
    images = np.stack(images)

    # keep article types with enough products, drop duplicate ids
    keep = ~df["id"].duplicated()
    counts = df.loc[keep, "articleType"].value_counts()
    keep &= df["articleType"].isin(counts[counts >= min_per_class].index)
    df, images = df[keep].reset_index(drop=True), images[keep.values]

    # stratified 70 / 15 / 15 split by article type
    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=df["articleType"])
    rest_y = df["articleType"].iloc[rest]
    ok = rest_y.value_counts().min() >= 2  # stratify only when every class can be split
    va, te = train_test_split(rest, test_size=0.50, random_state=SEED, stratify=rest_y if ok else None)
    df["split"] = "train"
    df.loc[va, "split"] = "val"
    df.loc[te, "split"] = "test"

    vocab = build_vocab(df.loc[df["split"] == "train", "productDisplayName"])  # train titles only: no leakage
    tokens = encode(df["productDisplayName"].tolist(), vocab)

    np.save(out / "images.npy", images)
    np.save(out / "tokens.npy", tokens)
    df.to_csv(out / "meta.csv", index=False)
    (out / "vocab.json").write_text(json.dumps(vocab))
    test_dir = out / "test_images"
    test_dir.mkdir(exist_ok=True)
    for pid in df.loc[df["split"] == "test", "id"]:
        (test_dir / f"{pid}.jpg").write_bytes(originals[pid])
    print(f"{len(df):,} products | {df['articleType'].nunique()} article types | vocab {len(vocab):,} words | "
          f"train {len(tr):,} / val {len(va):,} / test {len(te):,} -> {out}")
    return df


class ScratchData:
    """Loaded dataset with label arrays and split indices."""

    def __init__(self, root=SCRATCH_DIR):
        self.root = root
        self.meta = pd.read_csv(root / "meta.csv")
        self.images = np.load(root / "images.npy")
        self.tokens = np.load(root / "tokens.npy")
        self.vocab = json.loads((root / "vocab.json").read_text())
        m = self.meta
        self.y_type = pd.factorize(m["articleType"])[0]
        self.y_fine = pd.factorize(m["articleType"] + "|" + m["baseColour"].astype(str))[0]
        self.split = {s: np.flatnonzero(m["split"].values == s) for s in ("train", "val", "test")}

    def tokens_for(self, texts):
        return encode(texts, self.vocab)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "local", "folder"], default="hf")
    ap.add_argument("--path", default=None, help="dataset folder for --source folder")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-per-class", type=int, default=MIN_PER_CLASS)
    a = ap.parse_args()
    prepare(a.source, a.img_size, a.limit, a.min_per_class, path=a.path)
