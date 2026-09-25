"""Live online-shop catalog (https://dummyjson.com/products) embedded with OUR from-scratch model.

Only fashion categories are kept: the model was trained on fashion products and has no meaningful
notion of furniture, groceries or phones.

    python -m src.scratch.live     # fetch + embed (needs internet, about 1 minute)
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from ..live import API_URL, LIVE_DIR, _download_thumb, fetch_products, image_path
from . import serve
from .data import encode, square
from .train import to_input

FASHION = ["mens-shirts", "mens-shoes", "mens-watches", "sunglasses", "tops", "womens-bags", "womens-dresses",
           "womens-jewellery", "womens-shoes", "womens-watches", "fragrances"]
META = LIVE_DIR / "products_scratch.json"
EMB = serve.EMB_DIR / "live_gated.npy"


def title_of(p: dict) -> str:
    """Product text for the model: the shop title plus its category words ('womens bags')."""
    return f"{p['title']} {p['category'].replace('-', ' ')}"


@torch.no_grad()
def build(method: str = "gated", progress=None) -> dict:
    products = [p for p in fetch_products() if p["category"] in FASHION]
    kept, imgs = [], []
    for n, p in enumerate(products):
        im = _download_thumb(p)
        if im is not None:
            kept.append(p)
            imgs.append(square(im))
        if progress:
            progress((n + 1) / len(products), f"Downloading product photos {n + 1}/{len(products)}")
    model = serve.model(method)
    x = to_input(torch.from_numpy(np.stack(imgs)), False)
    t = torch.from_numpy(encode([title_of(p) for p in kept], serve.vocab())).long()
    f = model.encode(x, t)["f"].numpy().astype(np.float32)
    EMB.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB, f)
    META.write_text(json.dumps({"source": API_URL, "fetched_at": time.strftime("%Y-%m-%d %H:%M"),
                                "method": method, "products": kept}, indent=1))
    return load()


def load() -> dict | None:
    if not (META.exists() and EMB.exists()):
        return None
    cat = json.loads(META.read_text())
    cat["F"] = np.load(EMB)
    return cat


if __name__ == "__main__":
    c = build(progress=lambda f, m: print(f"\r{m}", end="", flush=True))
    print(f"\n{len(c['products'])} fashion products from {c['source']}")
