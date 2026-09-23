"""Checks for metrics, fusion scoring and every system x every query type (the app is tested in test_app.py).

    python -m pytest -q
Requires the cached catalog + embeddings (python -m src.data && python -m src.embeddings).
"""
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pytest
from PIL import Image

from src.config import IMAGE_DIR, RESULTS_DIR
from src.data import load_catalog
from src.evaluate import paired_bootstrap_p, ranking_metrics, topk_from_scores
from src.recommender import SYSTEMS, SYSTEMS_BY_NAME, Recommender

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = ("CLIP", "SigLIP")


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


# ---------------------------------------------------------------- every system x every query type (same path as app)
@pytest.fixture(scope="module")
def encoders():
    import pickle
    from src.config import EMB_DIR
    from src.encoders import get_encoder
    import gc
    cache = {}

    def get(name):
        if name not in cache:
            while len(cache) >= 2:  # keep at most 2 models in RAM, like the app
                cache.pop(next(iter(cache)))
                gc.collect()
            if name == "TF-IDF":
                with open(EMB_DIR / "tfidf.pkl", "rb") as f:
                    cache[name] = pickle.load(f)
            else:
                cache[name] = get_encoder(name)
        return cache[name]
    return get


def encode_query(enc, system, image=None, text=None):
    q = {}
    if image is not None:
        if system.image_model:
            q["q_img"] = enc(system.image_model).encode_image([image])[0]
        if system.text_model in SHARED:
            q["q_img_textenc"] = enc(system.text_model).encode_image([image])[0]
    if text:
        if system.text_model:
            q["q_txt"] = enc(system.text_model).encode_text([text])[0]
        if system.image_model in SHARED:
            q["q_txt_imgenc"] = enc(system.image_model).encode_text([text])[0]
    return q


@pytest.mark.parametrize("system", SYSTEMS, ids=lambda s: s.name)
def test_free_form_queries(system, df, encoders):
    r = Recommender(system, 0.5)
    img = Image.open(IMAGE_DIR / df.iloc[0]["image_path"])
    can_image = system.image_model is not None or system.text_model in SHARED
    can_text = system.text_model is not None or system.image_model in SHARED
    for image, text, supported in [(None, "black handbag for women", can_text),
                                   (img, None, can_image),
                                   (img, "in red colour", can_image or can_text)]:
        scores = r.query_scores(**encode_query(encoders, system, image, text))
        if not supported:
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
