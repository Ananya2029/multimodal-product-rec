"""REST API for the multimodal recommender (FastAPI).

    python -m uvicorn api:app --port 8000
    -> interactive docs at http://localhost:8000/docs

Endpoints
  GET  /health                              status + what is loaded
  GET  /systems                             available recommendation systems
  GET  /comparison                          model-comparison verdict (results/summary.json)
  GET  /products?q=&limit=&offset=          browse / filter the catalog
  GET  /products/{id}                       one product
  GET  /products/{id}/image                 product photo
  GET  /recommend/{id}?system=&k=&weight=   "more like this"
  GET  /search?q=&system=&k=&weight=        text search
  POST /search/image                        image upload search (multipart form field "file")
  POST /search/image-url                    image search from an internet URL, JSON {"url": "...", "text": "..."}
  GET  /live/search?q=&k=                   search the live internet catalog (dummyjson.com)
  GET  /gan/categories                      categories the GAN can draw
  GET  /gan/generate?category=&n=&seed=     GAN-generated product images (base64 PNG) + similar real products
"""
from __future__ import annotations

import base64
import io
import json
import os
from threading import Lock

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

from src.config import DEFAULT_IMAGE_WEIGHT, IMAGE_DIR, RESULTS_DIR
from src.data import load_catalog
from src.query import ModelCache, can_handle, encode_query, item_query
from src.recommender import SYSTEMS_BY_NAME, Recommender, available_systems
from src.web import ImageFetchError, fetch_image

DEFAULT_SYSTEM = "CLIP image + text"
MAX_UPLOAD = 10 * 1024 * 1024

app = FastAPI(title="Multimodal Product Recommender API",
              description="Product recommendations from image + text embeddings (CLIP, SigLIP, DINOv2, "
                          "MiniLM, TF-IDF, trained fusion heads) with a conditional GAN.",
              version="1.0")
df = load_catalog()
models = ModelCache(max_models=2)
lock = Lock()  # models are not thread-safe; FastAPI runs sync endpoints in a thread pool
_recs: dict = {}


# ------------------------------------------------------------------ helpers
def get_system(name: str):
    s = SYSTEMS_BY_NAME.get(name)
    if s is None or s not in available_systems():
        raise HTTPException(404, f"Unknown system '{name}'. See GET /systems.")
    return s


def get_rec(name: str, weight: float) -> Recommender:
    key = (name, round(weight, 2))
    if key not in _recs:
        _recs[key] = Recommender(get_system(name), weight)
    return _recs[key]


def product_json(i: int, score: float | None = None) -> dict:
    r = df.iloc[i]
    out = {"id": int(r["id"]), "name": r["productDisplayName"], "article_type": r["articleType"],
           "category": r["masterCategory"], "sub_category": r["subCategory"], "colour": r["baseColour"],
           "gender": r["gender"], "image_url": f"/products/{int(r['id'])}/image"}
    if score is not None:
        out["score"] = round(float(score), 4)
    return out


def index_of(product_id: int) -> int:
    hits = np.flatnonzero(df["id"].values == product_id)
    if not len(hits):
        raise HTTPException(404, f"No product with id {product_id}")
    return int(hits[0])


def run_query(system_name: str, k: int, weight: float, image=None, text=None, exclude=None) -> dict:
    s = get_system(system_name)
    if not can_handle(s, image is not None, bool(text and text.strip())):
        raise HTTPException(400, f"System '{system_name}' can't handle this query type "
                                 f"(image={image is not None}, text={bool(text)}).")
    rec = get_rec(system_name, weight)
    with lock:
        try:
            q = encode_query(s, image, text, models)
        except MemoryError as e:
            raise HTTPException(503, str(e))
    idx, sc = rec.top_k(rec.query_scores(**q), k, exclude=exclude)
    return {"system": system_name, "weight": rec.w, "results": [product_json(i, v) for i, v in zip(idx, sc)]}


def read_upload(data: bytes) -> Image.Image:
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image larger than 10 MB")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Uploaded file is not a readable image")


def png_b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    return {"status": "ok", "products": len(df), "loaded_models": list(models.models)}


@app.get("/systems")
def systems():
    return [{"name": s.name, "group": s.group, "image_model": s.image_model, "text_model": s.text_model,
             "trained": s.joint_model is not None} for s in available_systems()]


@app.get("/comparison")
def comparison():
    f = RESULTS_DIR / "summary.json"
    if not f.exists():
        raise HTTPException(404, "Run python -m src.evaluate first")
    out = json.loads(f.read_text())
    opt = RESULTS_DIR / "optimizer_comparison.csv"
    if opt.exists():
        import pandas as pd
        out["optimizers"] = pd.read_csv(opt).replace({np.nan: None}).to_dict("records")
    return out


@app.get("/products")
def products(q: str | None = None, limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0)):
    idx = np.arange(len(df))
    if q:
        idx = idx[df["productDisplayName"].str.contains(q, case=False, na=False).values]
    return {"total": int(len(idx)), "products": [product_json(i) for i in idx[offset:offset + limit]]}


@app.get("/products/{product_id}")
def product(product_id: int):
    return product_json(index_of(product_id))


@app.get("/products/{product_id}/image")
def product_image(product_id: int):
    return FileResponse(IMAGE_DIR / df.iloc[index_of(product_id)]["image_path"], media_type="image/jpeg")


@app.get("/recommend/{product_id}")
def recommend(product_id: int, system: str = DEFAULT_SYSTEM, k: int = Query(10, ge=1, le=100),
              weight: float = Query(DEFAULT_IMAGE_WEIGHT, ge=0, le=1)):
    i = index_of(product_id)
    rec = get_rec(system, weight)
    idx, sc = rec.top_k(rec.query_scores(**item_query(rec, i)), k, exclude=i)
    return {"query": product_json(i), "system": system, "weight": rec.w,
            "results": [product_json(j, v) for j, v in zip(idx, sc)]}


@app.get("/search")
def search(q: str = Query(..., min_length=1), system: str = DEFAULT_SYSTEM, k: int = Query(10, ge=1, le=100),
           weight: float = Query(DEFAULT_IMAGE_WEIGHT, ge=0, le=1)):
    return run_query(system, k, weight, text=q)


@app.post("/search/image")
async def search_image(file: UploadFile = File(...), text: str | None = None, system: str = DEFAULT_SYSTEM,
                       k: int = Query(10, ge=1, le=100), weight: float = Query(DEFAULT_IMAGE_WEIGHT, ge=0, le=1)):
    return run_query(system, k, weight, image=read_upload(await file.read()), text=text)


class UrlQuery(BaseModel):
    url: str
    text: str | None = None
    system: str = DEFAULT_SYSTEM
    k: int = 10
    weight: float = DEFAULT_IMAGE_WEIGHT


@app.post("/search/image-url")
def search_image_url(body: UrlQuery):
    try:
        img = fetch_image(body.url)
    except ImageFetchError as e:
        raise HTTPException(400, str(e))
    return run_query(body.system, max(1, min(body.k, 100)), min(max(body.weight, 0), 1), image=img, text=body.text)


@app.get("/live/search")
def live_search(q: str = Query(..., min_length=1), k: int = Query(10, ge=1, le=50)):
    from src.live import load_live_catalog, live_scores
    cat = load_live_catalog()
    if cat is None:
        raise HTTPException(404, "Live catalog not fetched yet: run python -m src.live (needs internet)")
    with lock:
        qv = models("CLIP").encode_text([q])[0]
    s = live_scores(cat, q_txt=qv)
    top = np.argsort(-s)[:k]
    return {"source": cat["source"], "fetched_at": cat["fetched_at"],
            "results": [{**{f: cat["products"][i].get(f) for f in ("id", "title", "category", "price", "thumbnail")},
                         "score": round(float(s[i]), 4)} for i in top]}


@app.get("/gan/categories")
def gan_categories():
    from src.gan import MODEL_DIR
    f = MODEL_DIR / "gan_meta.json"
    if not f.exists():
        raise HTTPException(404, "GAN not trained yet: run python -m src.gan")
    return json.loads(f.read_text())["classes"]


@app.get("/gan/generate")
def gan_generate(category: str, n: int = Query(4, ge=1, le=16), seed: int = 0, k: int = Query(5, ge=1, le=20)):
    from src.gan import generate, load_generator
    classes = gan_categories()
    if category not in classes:
        raise HTTPException(404, f"Unknown category. Choose one of {classes}")
    with lock:
        G, _ = load_generator()
        imgs = generate(G, classes.index(category), n, seed)
    similar = run_query(DEFAULT_SYSTEM, k, 1.0, image=imgs[0])["results"]  # image-only match for the first sample
    return {"category": category, "images_png_base64": [png_b64(im) for im in imgs],
            "similar_real_products_to_first_image": similar}
