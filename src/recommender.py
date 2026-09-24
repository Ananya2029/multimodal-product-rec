"""Recommendation systems = (image encoder, text encoder, fusion weight).

Scoring for a catalog item i:
    score_i = w * cos(q_img_space, I_i) + (1 - w) * cos(q_txt_space, T_i)

For vision-language models (CLIP/SigLIP) image and text share one space, so a text
query can also be matched against product images (and an image query against titles).

"Trained" systems use one joint embedding produced by the fusion head (src/fusion.py); it is
stored in the image slot (I) and queries arrive as q_img.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import embeddings


@dataclass(frozen=True)
class System:
    name: str
    image_model: str | None
    text_model: str | None
    group: str  # "text", "image", "multimodal" or "trained"
    joint_model: str | None = None  # trained fusion head, e.g. "Fusion-SGD"

    @property
    def shared(self):
        return self.image_model is not None and self.image_model == self.text_model


SYSTEMS = [
    # text only
    System("TF-IDF (title)", None, "TF-IDF", "text"),
    System("MiniLM (title)", None, "MiniLM", "text"),
    System("CLIP text", None, "CLIP", "text"),
    System("SigLIP text", None, "SigLIP", "text"),
    # image only
    System("ResNet50 image", "ResNet50", None, "image"),
    System("DINOv2 image", "DINOv2", None, "image"),
    System("CLIP image", "CLIP", None, "image"),
    System("SigLIP image", "SigLIP", None, "image"),
    # multimodal
    System("DINOv2 + MiniLM (late fusion)", "DINOv2", "MiniLM", "multimodal"),
    System("CLIP image + text", "CLIP", "CLIP", "multimodal"),
    System("SigLIP image + text", "SigLIP", "SigLIP", "multimodal"),
]
# trained fusion heads (python -m src.fusion); trained on 70% of the catalog, see results/optimizer_comparison.csv
TRAINED_SYSTEMS = [System(f"Trained fusion ({opt})", None, None, "trained", joint_model=f"Fusion-{opt}")
                   for opt in ("SGD", "Adam", "AdamW")]
SYSTEMS_BY_NAME = {s.name: s for s in SYSTEMS + TRAINED_SYSTEMS}


def available_systems() -> list[System]:
    """Zero-shot systems plus whichever trained fusion heads have been trained."""
    return SYSTEMS + [s for s in TRAINED_SYSTEMS if embeddings.emb_path(s.joint_model, "joint").exists()]


def _norm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


class Recommender:
    def __init__(self, system: System, weight: float = 0.5):
        self.s = system
        if system.joint_model:
            self.I, self.T = embeddings.load(system.joint_model, "joint"), None
        else:
            self.I = embeddings.load(system.image_model, "image") if system.image_model else None
            self.T = embeddings.load(system.text_model, "text") if system.text_model else None
        # single-modality systems ignore the weight
        self.w = 1.0 if self.T is None else 0.0 if self.I is None else weight

    # ---- item-to-item (all queries at once, used for evaluation)
    def item_scores(self) -> np.ndarray:
        S = 0
        if self.I is not None and self.w > 0:
            S = S + self.w * (self.I @ self.I.T)
        if self.T is not None and self.w < 1:
            S = S + (1 - self.w) * (self.T @ self.T.T)
        return S

    # ---- free-form query
    def query_scores(self, q_img: np.ndarray | None = None, q_txt: np.ndarray | None = None,
                     q_img_textenc: np.ndarray | None = None, q_txt_imgenc: np.ndarray | None = None):
        """
        q_img:          query image embedded with the image model
        q_txt:          query text embedded with the text model
        q_img_textenc:  query image embedded with the *text* model (only for shared-space models)
        q_txt_imgenc:   query text embedded with the *image* model (only for shared-space models)
        """
        parts, weights = [], []
        if self.I is not None:
            v = [x for x in (q_img, q_txt_imgenc) if x is not None]
            if v:
                parts.append(self.I @ _norm(np.sum(v, axis=0)))
                weights.append(self.w)
        if self.T is not None:
            v = [x for x in (q_txt, q_img_textenc) if x is not None]
            if v:
                parts.append(self.T @ _norm(np.sum(v, axis=0)))
                weights.append(1 - self.w)
        if not parts:
            return None
        weights = np.array(weights)
        weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(parts), 1 / len(parts))
        return sum(w * p for w, p in zip(weights, parts))

    @staticmethod
    def top_k(scores: np.ndarray, k: int = 10, exclude: int | None = None):
        s = scores.copy()
        if exclude is not None:
            s[exclude] = -np.inf
        idx = np.argpartition(-s, min(k, len(s) - 1))[:k]
        idx = idx[np.argsort(-s[idx])]
        return idx, s[idx]
