"""REST API tests (FastAPI TestClient, no server needed).

    python -m pytest -q tests/test_api.py
"""
import gc
import io
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(scope="module")
def client():
    import api
    with TestClient(api.app) as c:
        yield c
    api.models.models.clear()
    gc.collect()


def test_health_and_systems(client):
    assert client.get("/health").json()["status"] == "ok"
    names = [s["name"] for s in client.get("/systems").json()]
    assert "CLIP image + text" in names and "Trained fusion (SGD)" in names


def test_comparison(client):
    r = client.get("/comparison").json()
    assert r["recommended"] == "CLIP image + text" and len(r["optimizers"]) >= 5


def test_products_and_image(client):
    r = client.get("/products", params={"q": "watch", "limit": 3}).json()
    assert r["total"] > 0 and len(r["products"]) <= 3
    pid = r["products"][0]["id"]
    assert client.get(f"/products/{pid}").json()["id"] == pid
    img = client.get(f"/products/{pid}/image")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert client.get("/products/999999999").status_code == 404


@pytest.mark.parametrize("system", ["CLIP image + text", "Trained fusion (SGD)", "TF-IDF (title)"])
def test_recommend_and_search(client, system):
    pid = client.get("/products", params={"limit": 1}).json()["products"][0]["id"]
    r = client.get(f"/recommend/{pid}", params={"system": system, "k": 5}).json()
    assert len(r["results"]) == 5 and all(p["id"] != pid for p in r["results"])
    s = client.get("/search", params={"q": "black handbag for women", "system": system, "k": 5}).json()
    assert len(s["results"]) == 5


def test_image_upload(client):
    buf = io.BytesIO()
    Image.new("RGB", (80, 60), (20, 20, 20)).save(buf, format="PNG")
    r = client.post("/search/image", files={"file": ("x.png", buf.getvalue(), "image/png")}, params={"k": 3})
    assert r.status_code == 200 and len(r.json()["results"]) == 3
    bad = client.post("/search/image", files={"file": ("x.png", b"not an image", "image/png")})
    assert bad.status_code == 400


def test_errors_are_clean(client):
    assert client.get("/search", params={"q": "shoes", "system": "nope"}).status_code == 404
    # ResNet can't read text
    assert client.get("/search", params={"q": "shoes", "system": "ResNet50 image"}).status_code == 400
    r = client.post("/search/image-url", json={"url": "http://127.0.0.1/a.png"})
    assert r.status_code == 400 and "not allowed" in r.json()["detail"]


def test_gan_endpoint(client):
    cats = client.get("/gan/categories")
    if cats.status_code == 404:
        pytest.skip("GAN not trained")
    r = client.get("/gan/generate", params={"category": cats.json()[0], "n": 2, "k": 3}).json()
    assert len(r["images_png_base64"]) == 2 and len(r["similar_real_products_to_first_image"]) == 3
