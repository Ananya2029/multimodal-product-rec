"""Checks for metrics, fusion scoring and every system x every query type (the app is tested in test_app.py).

    python -m pytest -q
Requires the cached catalog + embeddings (python -m src.data && python -m src.embeddings).
"""
import json
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pytest
from PIL import Image

from src.config import IMAGE_DIR, RESULTS_DIR
from src.data import load_catalog
from src.evaluate import paired_bootstrap_p, ranking_metrics, topk_from_scores
from src import query as Q
from src.query import ModelCache
from src.recommender import SYSTEMS, SYSTEMS_BY_NAME, Recommender, available_systems

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def df():
    return load_catalog()


# ---------------------------------------------------------------- metrics
def test_perfect_and_empty_ranking():
    perfect = ranking_metrics(np.ones((2, 10)), np.array([10, 20]))
    assert perfect["NDCG@10"] == pytest.approx(1.0) and perfect["mAP@10"] == pytest.approx(1.0)
    none = ranking_metrics(np.zeros((2, 10)), np.array([5, 5]))
    assert none["NDCG@10"] == 0 and none["P@10"] == 0


def test_ndcg_prefers_relevant_items_higher():
    top = np.zeros((1, 10)); top[0, 0] = 1
    bottom = np.zeros((1, 10)); bottom[0, 9] = 1
    assert ranking_metrics(top, np.array([1]))["NDCG@10"] > ranking_metrics(bottom, np.array([1]))["NDCG@10"]


def test_topk_excludes_self_and_is_sorted():
    S = np.array([[9, 1, 5, 3], [2, 9, 8, 1], [1, 2, 9, 7]], float)
    top = topk_from_scores(S, k=2, exclude_self=True)
    assert top.tolist() == [[2, 3], [2, 0], [3, 1]]


def test_paired_bootstrap():
    a = np.ones(200)
    assert paired_bootstrap_p(a, a * 0) < 0.01          # clearly different
    assert paired_bootstrap_p(a, a.copy()) == 1.0       # identical


# ---------------------------------------------------------------- recommender
def test_embeddings_exist_and_are_normalised(df):
    for s in SYSTEMS:
        r = Recommender(s)
        for X in (r.I, r.T):
            if X is not None:
                assert X.shape[0] == len(df)
                assert np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-3)


def test_fusion_weight_extremes_match_single_modality():
    s = SYSTEMS_BY_NAME["CLIP image + text"]
    img_only = Recommender(SYSTEMS_BY_NAME["CLIP image"]).item_scores()
    txt_only = Recommender(SYSTEMS_BY_NAME["CLIP text"]).item_scores()
    assert np.allclose(Recommender(s, 1.0).item_scores(), img_only)
    assert np.allclose(Recommender(s, 0.0).item_scores(), txt_only)
    mid = Recommender(s, 0.5).item_scores()
    assert np.allclose(mid, 0.5 * img_only + 0.5 * txt_only)


def test_item_is_its_own_nearest_neighbour():
    r = Recommender(SYSTEMS_BY_NAME["CLIP image + text"])
    idx, _ = r.top_k(r.query_scores(q_img=r.I[7], q_txt=r.T[7]), 1)
    assert idx[0] == 7


# ---------------------------------------------------------------- every system x every query type (same code as app/API)
@pytest.fixture(scope="module")
def encoders():
    return ModelCache(max_models=2)  # keep at most 2 models in RAM, like the app


def encode_query(enc, system, image=None, text=None):
    return Q.encode_query(system, image, text, enc)


@pytest.mark.parametrize("system", available_systems(), ids=lambda s: s.name)
def test_free_form_queries(system, df, encoders):
    r = Recommender(system, 0.5)
    img = Image.open(IMAGE_DIR / df.iloc[0]["image_path"])
    for image, text in [(None, "black handbag for women"), (img, None), (img, "in red colour")]:
        try:
            q = encode_query(encoders, system, image, text)
        except MemoryError as e:  # low-RAM guard (src/encoders.py) refused to load a big model
            pytest.skip(str(e))
        scores = r.query_scores(**q)
        if not Q.can_handle(system, image is not None, bool(text)):
            assert scores is None
            continue
        assert scores is not None and scores.shape == (len(df),) and np.isfinite(scores).all()
        idx, s = r.top_k(scores, 10)
        assert len(set(idx)) == 10 and np.all(np.diff(s) <= 1e-6)


def test_text_search_is_relevant(df, encoders):
    r = Recommender(SYSTEMS_BY_NAME["CLIP image + text"])
    idx, _ = r.top_k(r.query_scores(**encode_query(encoders, SYSTEMS_BY_NAME["CLIP image + text"],
                                                   text="black handbag for women")), 10)
    assert (df.iloc[idx]["articleType"] == "Handbags").mean() >= 0.8


def test_uploaded_png_with_alpha_works(encoders, tmp_path):
    """Uploads may be RGBA PNGs or greyscale; the encoders must convert them."""
    rgba = Image.new("RGBA", (120, 90), (200, 30, 30, 128))
    grey = Image.new("L", (50, 50), 128)
    for im in (rgba, grey):
        v = encoders("CLIP").encode_image([im])
        assert v.shape == (1, 512) and np.isfinite(v).all()


# ---------------------------------------------------------------- results + app
def test_results_files_exist():
    for f in ("model_comparison.csv", "summary.json", "model_comparison.png", "fusion_sweep.png"):
        assert (RESULTS_DIR / f).exists(), f"missing {f}; run python -m src.evaluate"


# ---------------------------------------------------------------- trained fusion heads (optimizer comparison)
def test_optimizer_comparison_results():
    import pandas as pd
    res = pd.read_csv(RESULTS_DIR / "optimizer_comparison.csv")
    assert set(res["optimizer"]) >= {"SGD", "Adam", "AdamW"}
    zero_shot = res.iloc[0]["i2i NDCG@10"]
    for _, r in res[res["optimizer"] != "-"].iterrows():  # every trained head beats zero-shot on unseen products
        assert r["i2i NDCG@10"] > zero_shot


def test_trained_head_is_deterministic_and_normalised():
    from src.fusion import load_head, project
    head = load_head("SGD")
    v = np.random.default_rng(0).normal(size=(3, 512)).astype(np.float32)
    a, b = project(head, v, v), project(head, v, v)
    assert np.allclose(a, b) and np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-5)
    assert project(head, None, v).shape == (3, 256)  # text-only query works


# ---------------------------------------------------------------- GAN
def test_gan_generates_requested_images():
    from src.gan import MODEL_DIR, generate, load_generator
    if not (MODEL_DIR / "gan_generator.pt").exists():
        pytest.skip("GAN not trained (python -m src.gan)")
    G, classes = load_generator()
    imgs = generate(G, 0, 3, seed=1)
    assert len(imgs) == 3 and imgs[0].size == (64, 64)
    assert np.asarray(generate(G, 0, 1, seed=1)[0]).tolist() == np.asarray(imgs[0]).tolist()  # seeded
    metrics = json.loads((RESULTS_DIR / "gan_metrics.json").read_text())
    assert metrics["class accuracy (generated)"] > metrics["chance accuracy"]


# ---------------------------------------------------------------- internet
@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://127.0.0.1:8000/x.png", "http://localhost/a.jpg",
                                 "http://192.168.1.1/a.jpg", "ftp://example.com/a.jpg", "not a url"])
def test_url_fetch_refuses_unsafe_links(url):
    from src.web import ImageFetchError, fetch_image
    with pytest.raises(ImageFetchError):
        fetch_image(url)
