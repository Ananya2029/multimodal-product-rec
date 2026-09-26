"""Load the exported from-scratch models (models/scratch/export.json) and answer queries on CPU.

The catalog is the TEST split: products that no model saw during training.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ..config import ROOT
from .data import SCRATCH_DIR, encode, square
from .models import MultimodalRec
from .train import to_input

MODEL_DIR = ROOT / "models" / "scratch"
EMB_DIR = ROOT / "embeddings" / "scratch"
RES_DIR = ROOT / "results" / "scratch"


def available() -> bool:
    return (MODEL_DIR / "export.json").exists() and (EMB_DIR / "test_meta.csv").exists()


@lru_cache(maxsize=1)
def export_info() -> dict:
    return json.loads((MODEL_DIR / "export.json").read_text())


@lru_cache(maxsize=1)
def catalog() -> pd.DataFrame:
    df = pd.read_csv(EMB_DIR / "test_meta.csv").reset_index(drop=True)
    df["image_file"] = [str(SCRATCH_DIR / "test_images" / f"{i}.jpg") for i in df["id"]]
    return df


@lru_cache(maxsize=1)
def vocab() -> dict:
    return json.loads((SCRATCH_DIR / "vocab.json").read_text())


@lru_cache(maxsize=None)
def item_embeddings(method: str) -> np.ndarray:
    return np.load(EMB_DIR / f"{method}.npy")


@lru_cache(maxsize=None)
def model(method: str) -> MultimodalRec:
    info = export_info()["methods"][method]
    m = MultimodalRec(info.get("arch", method), len(vocab()))  # extension variants reuse an architecture
    m.load_state_dict(torch.load(MODEL_DIR / info["checkpoint"], map_location="cpu"))
    return m.eval()


def can_handle(method: str, image: bool, text: bool) -> bool:
    method = export_info()["methods"].get(method, {}).get("arch", method)
    if method == "image":
        return image
    if method == "text":
        return text
    return image or text


@torch.no_grad()
def encode_query(method: str, image: Image.Image | None = None, text: str | None = None) -> np.ndarray | None:
    m = model(method)
    text = (text or "").strip() or None
    x = to_input(torch.from_numpy(square(image)).unsqueeze(0), False) if image is not None and m.use_img else None
    t = torch.from_numpy(encode([text], vocab())).long() if text and m.use_txt else None
    if x is None and t is None:
        return None
    return m.encode(x, t)["f"][0].numpy()


def scores(method: str, q: np.ndarray) -> np.ndarray:
    return item_embeddings(method) @ q


def item_query(method: str, i: int) -> np.ndarray:
    return item_embeddings(method)[i]


def top_k(s: np.ndarray, k: int, exclude: int | None = None):
    s = s.astype(float).copy()
    if exclude is not None:
        s[exclude] = -np.inf
    idx = np.argsort(-s)[:k]
    return idx, s[idx]
