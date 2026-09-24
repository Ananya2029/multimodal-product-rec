"""Turn a free-form query (image and/or text) into the vectors Recommender.query_scores expects.

Shared by the Streamlit app, the REST API and the tests, so all three behave identically.
`get_model(name)` must return a loaded model; callers decide how to cache (see ModelCache).
"""
from __future__ import annotations

import gc
import pickle
from collections import OrderedDict
from typing import Callable

import numpy as np
from PIL import Image

from .config import EMB_DIR
from .recommender import System

SHARED = ("CLIP", "SigLIP")  # models whose image and text vectors live in one space


def load_model(name: str):
    """Load an encoder ("TF-IDF", "CLIP", ...) or a trained fusion head ("Fusion-SGD", ...)."""
    if name == "TF-IDF":
        with open(EMB_DIR / "tfidf.pkl", "rb") as f:
            return pickle.load(f)
    if name.startswith("Fusion-"):
        from .fusion import load_head
        return load_head(name.removeprefix("Fusion-"))
    from .encoders import get_encoder
    return get_encoder(name)


class ModelCache:
    """Keeps at most `max_models` models in RAM (least recently used ones are unloaded)."""

    def __init__(self, max_models: int = 2):
        self.max, self.models = max_models, OrderedDict()

    def __call__(self, name: str):
        if name in self.models:
            self.models.move_to_end(name)
        else:
            while len(self.models) >= self.max:
                self.models.popitem(last=False)
                gc.collect()
            self.models[name] = load_model(name)
        return self.models[name]


def can_handle(system: System, image: bool, text: bool) -> bool:
    if system.joint_model:
        return image or text
    ok_img = system.image_model is not None or system.text_model in SHARED
    ok_txt = system.text_model is not None or system.image_model in SHARED
    return (image and ok_img) or (text and ok_txt)


def encode_query(system: System, image: Image.Image | None, text: str | None,
                 get_model: Callable[[str], object]) -> dict:
    """Embed the query for every side of the catalog the system can score."""
    text = (text or "").strip() or None
    q = {}
    if system.joint_model:  # trained fusion head on top of CLIP
        from .fusion import project
        clip = get_model("CLIP")
        img_vec = clip.encode_image([image])[0] if image is not None else None
        txt_vec = clip.encode_text([text])[0] if text else None
        if img_vec is None and txt_vec is None:
            return q
        q["q_img"] = project(get_model(system.joint_model), img_vec, txt_vec)[0]
        return q
    if image is not None:
        if system.image_model:
            q["q_img"] = get_model(system.image_model).encode_image([image])[0]
        if system.text_model in SHARED:
            q["q_img_textenc"] = get_model(system.text_model).encode_image([image])[0]
    if text:
        if system.text_model:
            q["q_txt"] = get_model(system.text_model).encode_text([text])[0]
        if system.image_model in SHARED:
            q["q_txt_imgenc"] = get_model(system.image_model).encode_text([text])[0]
    return q


def item_query(rec, i: int) -> dict:
    """Query vectors for catalog item i ("more like this"), from the cached embeddings."""
    q = {}
    if rec.I is not None:
        q["q_img"] = np.asarray(rec.I[i])
    if rec.T is not None:
        q["q_txt"] = np.asarray(rec.T[i])
    return q
