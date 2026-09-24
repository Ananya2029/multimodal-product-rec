"""Image / text encoders behind one interface.

Every encoder returns L2-normalised float32 numpy arrays, so cosine similarity == dot product.
`encode_text` / `encode_image` return None when a modality is not supported.
"""
from __future__ import annotations

import os

# Load model weights sequentially: transformers 5 otherwise reads them on 4 threads at once, which raises
# peak memory and crashed (access violation) on a low-RAM Windows machine inside Streamlit.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .config import BATCH_SIZE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# NB: every torch encode_* method is wrapped in @torch.no_grad(). A global torch.set_grad_enabled(False)
# would only apply to the importing thread, and Streamlit reruns scripts in new threads.


def _l2(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = F.normalize(x.detach().float(), dim=-1).cpu().numpy()
    else:
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return x.astype(np.float32)


def _as_tensor(out) -> torch.Tensor:
    """transformers v5 get_*_features may return a ModelOutput instead of a tensor."""
    if isinstance(out, torch.Tensor):
        return out
    return out.pooler_output


# Free RAM needed before loading one tower. Below this, loading a memory-mapped checkpoint on Windows can
# die with an access violation (not a Python error), which would kill the whole app, so refuse politely.
MIN_FREE_MB = {"SigLIP": 1200, "CLIP": 600, "DINOv2": 500, "ResNet50": 500, "MiniLM": 400}


def _check_free_memory(model_name: str):
    need = MIN_FREE_MB.get(model_name)
    if not need:
        return
    import psutil
    free = psutil.virtual_memory().available / 2**20
    if free < need:
        raise MemoryError(f"Not enough free memory to load {model_name} ({free:.0f} MB free, about {need} MB "
                          f"needed). Close other programs (VS Code, Chrome tabs, Docker, WhatsApp) and try again, "
                          f"or choose a lighter model such as TF-IDF (title).")


def _batches(items, bs, desc):
    for i in tqdm(range(0, len(items), bs), desc=desc, leave=False):
        yield items[i:i + bs]


def _load_images(paths):
    return [Image.open(p).convert("RGB") if not isinstance(p, Image.Image) else p.convert("RGB")
            for p in paths]


class Encoder:
    name: str = ""
    modalities: tuple[str, ...] = ()
    shared_space = False  # True when text and image live in the same space (CLIP-style)

    def encode_text(self, texts: list[str]) -> np.ndarray | None:
        return None

    def encode_image(self, images) -> np.ndarray | None:
        return None


# ---------------------------------------------------------------- text-only
class TfidfEncoder(Encoder):
    """Classic lexical baseline. Must be fit on the catalog text first."""
    name, modalities = "TF-IDF", ("text",)

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.fitted = False

    def fit(self, texts):
        self.vec.fit(texts)
        self.fitted = True
        return self

    def encode_text(self, texts):
        return _l2(self.vec.transform(texts).toarray())


class MiniLMEncoder(Encoder):
    """sentence-transformers/all-MiniLM-L6-v2 with mean pooling (no extra dependency)."""
    name, modalities = "MiniLM", ("text",)
    model_id = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        _check_free_memory("MiniLM")
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(DEVICE).eval()

    @torch.no_grad()
    def encode_text(self, texts):
        out = []
        for b in _batches(texts, 128, "MiniLM text"):
            t = self.tok(b, padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
            h = self.model(**t).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1).float()
            out.append(_l2((h * m).sum(1) / m.sum(1)))
        return np.vstack(out)


# ---------------------------------------------------------------- image-only
class ResNetEncoder(Encoder):
    """ImageNet ResNet-50, penultimate layer (2048-d)."""
    name, modalities = "ResNet50", ("image",)

    def __init__(self):
        _check_free_memory("ResNet50")
        from torchvision.models import ResNet50_Weights, resnet50
        w = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=w)
        self.model.fc = torch.nn.Identity()
        self.model = self.model.to(DEVICE).eval()
        self.tf = w.transforms()

    @torch.no_grad()
    def encode_image(self, images):
        out = []
        for b in _batches(images, BATCH_SIZE, "ResNet image"):
            x = torch.stack([self.tf(im) for im in _load_images(b)]).to(DEVICE)
            out.append(_l2(self.model(x)))
        return np.vstack(out)


class DinoV2Encoder(Encoder):
    """Self-supervised DINOv2 ViT-S/14, CLS token (384-d)."""
    name, modalities = "DINOv2", ("image",)
    model_id = "facebook/dinov2-small"

    def __init__(self):
        _check_free_memory("DINOv2")
        from transformers import AutoImageProcessor, AutoModel
        self.proc = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(DEVICE).eval()

    @torch.no_grad()
    def encode_image(self, images):
        out = []
        for b in _batches(images, BATCH_SIZE, "DINOv2 image"):
            x = self.proc(images=_load_images(b), return_tensors="pt").to(DEVICE)
            out.append(_l2(self.model(**x).pooler_output))
        return np.vstack(out)


# ---------------------------------------------------------------- vision-language (shared space)
class CLIPEncoder(Encoder):
    """Vision-language model. The text and image towers are loaded lazily, each on first use: a text
    search only needs the text tower, which roughly halves memory on low-RAM machines."""
    name, modalities, shared_space = "CLIP", ("image", "text"), True
    model_id = "openai/clip-vit-base-patch32"
    text_padding: str | bool = True
    text_cls, vision_cls = "CLIPTextModelWithProjection", "CLIPVisionModelWithProjection"

    def __init__(self):
        from transformers import AutoProcessor
        self.proc = AutoProcessor.from_pretrained(self.model_id)
        self._towers = {}

    def _tower(self, cls_name: str):
        if cls_name not in self._towers:
            import transformers
            _check_free_memory(self.name)
            self._towers[cls_name] = getattr(transformers, cls_name).from_pretrained(self.model_id).to(DEVICE).eval()
        return self._towers[cls_name]

    @staticmethod
    def _embeds(out, key):
        return getattr(out, key, None) if getattr(out, key, None) is not None else out.pooler_output

    @torch.no_grad()
    def encode_text(self, texts):
        tower, out = self._tower(self.text_cls), []
        for b in _batches(texts, 128, f"{self.name} text"):
            t = self.proc(text=b, padding=self.text_padding, truncation=True,
                          max_length=64, return_tensors="pt").to(DEVICE)
            t = {k: v for k, v in t.items() if k in ("input_ids", "attention_mask")}
            out.append(_l2(self._embeds(tower(**t), "text_embeds")))
        return np.vstack(out)

    @torch.no_grad()
    def encode_image(self, images):
        tower, out = self._tower(self.vision_cls), []
        for b in _batches(images, BATCH_SIZE, f"{self.name} image"):
            x = self.proc(images=_load_images(b), return_tensors="pt").to(DEVICE)
            out.append(_l2(self._embeds(tower(pixel_values=x["pixel_values"]), "image_embeds")))
        return np.vstack(out)


class SigLIPEncoder(CLIPEncoder):
    name = "SigLIP"
    model_id = "google/siglip-base-patch16-224"
    text_padding = "max_length"  # SigLIP was trained with max-length padding
    text_cls, vision_cls = "SiglipTextModel", "SiglipVisionModel"  # pooler_output = the embedding


ENCODERS = {
    "TF-IDF": TfidfEncoder,
    "MiniLM": MiniLMEncoder,
    "ResNet50": ResNetEncoder,
    "DINOv2": DinoV2Encoder,
    "CLIP": CLIPEncoder,
    "SigLIP": SigLIPEncoder,
}


def get_encoder(name: str) -> Encoder:
    return ENCODERS[name]()
