"""
fusion.py
Combines image and text embeddings into one item representation.
Supports 4 modes for the ablation study: image_only, text_only, concat, gated.

Usage:
    python src/fusion.py --config configs/default.yaml --mode gated
"""
import argparse
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class GatedFusion(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_img, z_txt):
        alpha = torch.sigmoid(self.gate(torch.cat([z_img, z_txt], dim=-1)))
        z_fused = alpha * z_img + (1 - alpha) * z_txt
        return self.norm(self.proj(z_fused)), alpha


class ConcatFusion(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_img, z_txt):
        z = torch.cat([z_img, z_txt], dim=-1)
        return self.norm(self.proj(z)), None


def load_aligned_embeddings(processed_dir):
    """Loads image + text embeddings and aligns them on the shared item (ASIN) set."""
    img_mat = np.load(os.path.join(processed_dir, "image_embeddings.npy"))
    with open(os.path.join(processed_dir, "image_embeddings_asins.txt")) as f:
        img_asins = f.read().splitlines()
    txt_mat = np.load(os.path.join(processed_dir, "text_embeddings.npy"))
    with open(os.path.join(processed_dir, "text_embeddings_asins.txt")) as f:
        txt_asins = f.read().splitlines()

    img_dict = dict(zip(img_asins, img_mat))
    txt_dict = dict(zip(txt_asins, txt_mat))
    shared = [a for a in txt_asins if a in img_dict]  # only items with BOTH modalities can use gated/concat

    img_aligned = np.stack([img_dict[a] for a in shared])
    txt_aligned = np.stack([txt_dict[a] for a in shared])
    return shared, img_aligned, txt_aligned, img_dict, txt_dict


def build_fused(mode, processed_dir, dim=256, out_name=None, save=True):
    shared, img_aligned, txt_aligned, img_dict, txt_dict = load_aligned_embeddings(processed_dir)

    if mode == "image_only":
        asins, mat = list(img_dict.keys()), np.stack(list(img_dict.values()))
    elif mode == "text_only":
        asins, mat = list(txt_dict.keys()), np.stack(list(txt_dict.values()))
    else:
        z_img = torch.tensor(img_aligned, dtype=torch.float32)
        z_txt = torch.tensor(txt_aligned, dtype=torch.float32)
        fusion = GatedFusion(dim) if mode == "gated" else ConcatFusion(dim)
        fusion.eval()
        with torch.no_grad():
            z_fused, alpha = fusion(z_img, z_txt)
        mat = F.normalize(z_fused, dim=-1).numpy()
        asins = shared

    if save:
        out_name = out_name or f"fused_{mode}_embeddings.npy"
        np.save(os.path.join(processed_dir, out_name), mat)
        with open(os.path.join(processed_dir, out_name.replace(".npy", "_asins.txt")), "w") as f:
            f.write("\n".join(asins))
        print(f"[{mode}] saved {len(asins)} embeddings -> {out_name}")

    return asins, mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["image_only", "text_only", "concat", "gated"], default="gated")
    args = parser.parse_args()
    cfg = load_config(args.config)
    build_fused(args.mode, cfg["data"]["processed_dir"], cfg["fusion"]["dim"])
