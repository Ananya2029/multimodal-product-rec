"""Live product catalog from a public shop API on the internet (https://dummyjson.com/products).

Products (title, category, price, thumbnail) are downloaded, embedded with CLIP (image + text) and
cached on disk. Because CLIP puts every product in one shared space, live products can be matched
against each other, against free text, and against our fashion catalog.

    python -m src.live   # fetch + embed (about 1-2 min on a CPU)
"""
from __future__ import annotations

import io
import json
import time

import numpy as np
import requests
from PIL import Image

from .config import DATA_DIR, EMB_DIR

API_URL = "https://dummyjson.com/products"
LIVE_DIR = DATA_DIR / "live"
IMG_DIR = LIVE_DIR / "images"
CATALOG_JSON = LIVE_DIR / "products.json"
FIELDS = "id,title,category,price,rating,brand,thumbnail,description"


def fetch_products(timeout=20) -> list[dict]:
    r = requests.get(API_URL, params={"limit": 0, "select": FIELDS}, timeout=timeout)
    r.raise_for_status()
    return r.json()["products"]


def _download_thumb(p: dict) -> Image.Image | None:
    path = IMG_DIR / f"{p['id']}.png"
    if path.exists():
        return Image.open(path).convert("RGB")
    try:
        r = requests.get(p["thumbnail"], timeout=20)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        im.save(path)
        return im
    except Exception:
        return None


def build_live_catalog(clip=None, progress=None) -> dict:
    """Download products + thumbnails from the API and embed them with CLIP. Returns the loaded catalog."""
    from .encoders import CLIPEncoder

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    products = fetch_products()
    kept, images = [], []
    for n, p in enumerate(products):
        im = _download_thumb(p)
        if im is not None:
            kept.append(p)
            images.append(im)
        if progress:
            progress((n + 1) / len(products), f"Downloading images {n + 1}/{len(products)}")
    clip = clip or CLIPEncoder()
    if progress:
        progress(1.0, "Embedding products with CLIP ...")
    texts = [f"{p['title']} ({p['category'].replace('-', ' ')})" for p in kept]
    np.save(EMB_DIR / "live_CLIP_image.npy", clip.encode_image(images))
    np.save(EMB_DIR / "live_CLIP_text.npy", clip.encode_text(texts))
    meta = {"source": API_URL, "fetched_at": time.strftime("%Y-%m-%d %H:%M"), "products": kept}
    CATALOG_JSON.write_text(json.dumps(meta, indent=1))
    return load_live_catalog()


def load_live_catalog() -> dict | None:
    """Cached live catalog, or None if it has not been fetched yet."""
    fi, ft = EMB_DIR / "live_CLIP_image.npy", EMB_DIR / "live_CLIP_text.npy"
    if not (CATALOG_JSON.exists() and fi.exists() and ft.exists()):
        return None
    meta = json.loads(CATALOG_JSON.read_text())
    meta["I"], meta["T"] = np.load(fi), np.load(ft)
    return meta


def image_path(p: dict):
    return IMG_DIR / f"{p['id']}.png"


def live_scores(cat: dict, q_img=None, q_txt=None, weight=0.5) -> np.ndarray:
    """CLIP image+text fusion over the live catalog. q_img / q_txt are CLIP vectors (either may be None)."""
    q = [v for v in (q_img, q_txt) if v is not None]
    q = np.sum(q, axis=0)
    q = q / (np.linalg.norm(q) + 1e-12)
    return weight * (cat["I"] @ q) + (1 - weight) * (cat["T"] @ q)


if __name__ == "__main__":
    cat = build_live_catalog(progress=lambda f, msg: print(f"\r{msg}", end="", flush=True))
    print(f"\n{len(cat['products'])} live products from {cat['source']} "
          f"({len({p['category'] for p in cat['products']})} categories)")
