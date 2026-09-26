"""Encoders and fusion methods, all randomly initialised (no pretrained weights).

Image encoder   ResNet-style CNN: 64x64x3 -> 8x8x256 feature map -> pooled 256-d feature
Text encoder    word embeddings + learned positions + 2-layer Transformer -> pooled 256-d feature

Methods (how the item embedding f used for retrieval is built):
    image   f = z_img                          image-only baseline
    text    f = z_txt                          text-only baseline
    early   f = MLP([h_img ; h_txt])           early fusion (feature concatenation)
    late    f = [sqrt(w) z_img ; sqrt(1-w) z_txt]  late fusion (weighted score sum, w = 0.5)
    gated   g = sigmoid(W [h_img ; h_txt]),  f = MLP(g * h_img + (1 - g) * h_txt)
            gated multimodal unit: per-dimension, per-item weighting of the modalities   (proposed)
    xattn   title tokens attend over the 8x8 image regions (cross-attention), then MLP   (proposed)

Every method also produces per-modality embeddings z_img, z_txt in one shared space (aligned with an
image-text contrastive loss), so a text-only or image-only query can still be answered. A missing
modality is replaced by a learned "null" vector, and training randomly drops modalities
(modality dropout) so the fused models learn to handle that. A missing-modality consistency loss
additionally pulls the text-only and image-only fused embeddings of an item towards its full
(image + text) embedding, so single-modality queries land next to the right products.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

METHODS = ["image", "text", "early", "late", "gated", "xattn"]
METHOD_NAMES = {
    "image": "Image-only CNN",
    "text": "Text-only Transformer",
    "early": "Early fusion (concat)",
    "late": "Late fusion (score sum)",
    "gated": "Gated fusion (proposed)",
    "xattn": "Cross-attention fusion (proposed)",
    # extension variants of the gated model (same architecture, different training)
    "gated_uni": "Gated fusion + uni-modal supervision (proposed, final)",
    "gated_noise_uni": "Gated fusion + noise-aware + uni-modal (robust variant)",
}
FEAT = 256   # encoder feature size
EMB = 128    # retrieval embedding size


# ------------------------------------------------------------------ encoders
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                      if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(h)) + self.short(x))


class ImageCNN(nn.Module):
    """ResNet-style CNN trained from scratch. 64x64 input -> (B, 256, 8, 8)."""

    def __init__(self, widths=(32, 64, 128, 256)):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.stem = nn.Sequential(nn.Conv2d(3, w0, 3, 1, 1, bias=False), nn.BatchNorm2d(w0), nn.ReLU(True))
        self.stages = nn.Sequential(
            BasicBlock(w0, w1, 2), BasicBlock(w1, w1),
            BasicBlock(w1, w2, 2), BasicBlock(w2, w2),
            BasicBlock(w2, w3, 2), BasicBlock(w3, w3))
        self.out_dim = w3

    def forward(self, x):
        fmap = self.stages(self.stem(x))            # (B, 256, 8, 8)
        tokens = fmap.flatten(2).transpose(1, 2)    # (B, 64, 256) region features for cross-attention
        return fmap.mean((2, 3)), tokens


class TextTransformer(nn.Module):
    """Word embeddings + positions + Transformer encoder, trained from scratch on product titles."""

    def __init__(self, vocab_size, max_len=16, d=128, layers=2, heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, FEAT)
        self.out_dim = FEAT

    def forward(self, ids):
        pad = ids == 0
        pad[:, 0] = pad[:, 0] & ~pad.all(1)          # an all-empty title keeps one position to attend to
        h = self.enc(self.emb(ids) + self.pos[:, :ids.shape[1]], src_key_padding_mask=pad)
        h = self.norm(h)
        keep = (~pad).unsqueeze(-1).float()
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1)
        return self.out(pooled), h, pad


# ------------------------------------------------------------------ full model
class MultimodalRec(nn.Module):
    def __init__(self, method: str, vocab_size: int, max_len: int = 16, late_w: float = 0.5):
        super().__init__()
        assert method in METHODS, method
        self.method, self.late_w = method, late_w
        self.use_img = method != "text"
        self.use_txt = method != "image"
        if self.use_img:
            self.img = ImageCNN()
            self.proj_i = nn.Linear(FEAT, EMB)
        if self.use_txt:
            self.txt = TextTransformer(vocab_size, max_len)
            self.proj_t = nn.Linear(FEAT, EMB)
        if method in ("early", "gated", "xattn"):
            self.null_i = nn.Parameter(torch.zeros(FEAT))   # learned stand-in for a missing modality
            self.null_t = nn.Parameter(torch.zeros(FEAT))
        if method == "early":
            self.fuse = nn.Sequential(nn.Linear(2 * FEAT, 2 * FEAT), nn.GELU(), nn.Linear(2 * FEAT, EMB))
        if method == "gated":
            self.gate = nn.Linear(2 * FEAT, FEAT)
            self.fuse = nn.Sequential(nn.LayerNorm(FEAT), nn.Linear(FEAT, FEAT), nn.GELU(), nn.Linear(FEAT, EMB))
        if method == "xattn":
            self.q = nn.Linear(128, FEAT)
            self.attn = nn.MultiheadAttention(FEAT, 4, batch_first=True)
            self.null_tok_i = nn.Parameter(torch.zeros(1, 1, FEAT))
            self.null_tok_t = nn.Parameter(torch.zeros(1, 1, FEAT))
            self.fuse = nn.Sequential(nn.LayerNorm(3 * FEAT), nn.Linear(3 * FEAT, 2 * FEAT), nn.GELU(),
                                      nn.Linear(2 * FEAT, EMB))
        # learnable temperature for the image-text contrastive loss (CLIP-style), init 1/0.07
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    # ---- helpers
    def _drop_masks(self, b, device, p):
        """Modality dropout: per item, drop the image OR the text with probability p/2 each."""
        r = torch.rand(b, device=device)
        return r < p / 2, (r >= p / 2) & (r < p)   # (drop_img, drop_txt)

    def encode(self, images=None, tokens=None, drop_p: float = 0.0, consistency: bool = False,
               img_out=None) -> dict:
        """images: (B,3,64,64) float or None; tokens: (B,L) long or None. Returns f, z_img, z_txt, gate.
        consistency=True also returns f_full / f_img_only / f_txt_only for the consistency loss.
        img_out: precomputed self.img(images) output, to re-use image features across many titles."""
        ref = images if images is not None else (img_out[0] if img_out is not None else tokens)
        b, dev = ref.shape[0], ref.device
        out = {"z_img": None, "z_txt": None, "gate": None}
        h_i = tok_i = h_t = tok_t = pad_t = None
        if self.use_img and (images is not None or img_out is not None):
            h_i, tok_i = img_out if img_out is not None else self.img(images)
            out["z_img"] = F.normalize(self.proj_i(h_i), dim=-1)
        if self.use_txt and tokens is not None:
            h_t, tok_t, pad_t = self.txt(tokens)
            out["z_txt"] = F.normalize(self.proj_t(h_t), dim=-1)

        m = self.method
        if m == "image":
            out["f"] = out["z_img"]
            return out
        if m == "text":
            out["f"] = out["z_txt"]
            return out
        if m == "late":
            zi = out["z_img"] if out["z_img"] is not None else out["z_txt"]   # shared space fallback
            zt = out["z_txt"] if out["z_txt"] is not None else out["z_img"]
            out["f"] = torch.cat([math.sqrt(self.late_w) * zi, math.sqrt(1 - self.late_w) * zt], -1)
            return out

        # early / gated / xattn: substitute learned null vectors for missing or dropped modalities
        drop_i, drop_t = self._drop_masks(b, dev, drop_p) if drop_p > 0 else (None, None)
        out["f"], out["gate"] = self._fuse(b, dev, h_i, tok_i, h_t, tok_t, pad_t, drop_i, drop_t)
        if consistency and h_i is not None and h_t is not None:
            # the same items seen with both modalities (target) and with only one of them
            out["f_full"], _ = self._fuse(b, dev, h_i, tok_i, h_t, tok_t, pad_t)
            out["f_img_only"], _ = self._fuse(b, dev, h_i, tok_i, None, None, None)
            out["f_txt_only"], _ = self._fuse(b, dev, None, None, h_t, tok_t, pad_t)
        return out

    def _fuse(self, b, dev, h_i, tok_i, h_t, tok_t, pad_t, drop_i=None, drop_t=None):
        """Fused, normalised embedding from encoder outputs; None / dropped modalities use learned nulls."""
        m = self.method
        hi = self.null_i.expand(b, -1) if h_i is None else h_i
        ht = self.null_t.expand(b, -1) if h_t is None else h_t
        if drop_i is not None:
            hi = torch.where(drop_i[:, None], self.null_i.expand(b, -1), hi)
            ht = torch.where(drop_t[:, None], self.null_t.expand(b, -1), ht)
        g = None
        if m == "early":
            f = self.fuse(torch.cat([hi, ht], -1))
        elif m == "gated":
            g = torch.sigmoid(self.gate(torch.cat([hi, ht], -1)))
            f = self.fuse(g * hi + (1 - g) * ht)
        else:  # xattn: title tokens (queries) attend over image regions (keys / values)
            kv = self.null_tok_i.expand(b, 1, -1) if tok_i is None else tok_i
            if tok_t is None:
                qtok, qpad = self.null_tok_t.expand(b, 1, -1), torch.zeros(b, 1, dtype=torch.bool, device=dev)
            else:
                qtok, qpad = self.q(tok_t), pad_t
            if drop_i is not None:
                kv = torch.where(drop_i[:, None, None], self.null_tok_i.expand_as(kv), kv)
                qtok = torch.where(drop_t[:, None, None], self.null_tok_t.expand_as(qtok), qtok)
            att, _ = self.attn(qtok, kv, kv, need_weights=False)
            keep = (~qpad).unsqueeze(-1).float()
            att = (att * keep).sum(1) / keep.sum(1).clamp(min=1)
            # pooled image + attended regions + pooled title: a text-only query still carries its text
            f = self.fuse(torch.cat([hi, att, ht], -1))
        return F.normalize(f, dim=-1), g


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
