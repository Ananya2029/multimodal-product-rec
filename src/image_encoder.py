"""
image_encoder.py
Loads a pretrained SigLIP checkpoint and (optionally) LoRA-fine-tunes the
projection head with contrastive loss on (image, title) pairs.
Outputs a 256-d embedding per item, cached to disk as a .npy matrix.

Usage:
    python src/image_encoder.py --config configs/default.yaml
"""
import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model
from tqdm import tqdm


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class ImageTitleDataset(Dataset):
    def __init__(self, item_df, processor):
        self.df = item_df[item_df["image_path"].notna()].reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return image, row["title"]


def collate_fn(batch, processor):
    images, titles = zip(*batch)
    inputs = processor(
        text=list(titles), images=list(images), padding="max_length",
        truncation=True, return_tensors="pt"
    )
    return inputs


class SiglipEncoder(nn.Module):
    def __init__(self, model_name, embed_dim, use_lora=True, lora_r=8, lora_alpha=16):
        super().__init__()
        base = AutoModel.from_pretrained(model_name)
        if use_lora:
            lora_cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05, bias="none",
            )
            base = get_peft_model(base, lora_cfg)
        self.model = base
        hidden = base.config.vision_config.hidden_size if hasattr(base, "config") else 768
        self.img_proj = nn.Linear(hidden, embed_dim)

    def forward(self, pixel_values, input_ids=None, attention_mask=None):
        out = self.model(pixel_values=pixel_values, input_ids=input_ids,
                          attention_mask=attention_mask)
        img_emb = out.image_embeds if hasattr(out, "image_embeds") else out.vision_model_output.pooler_output
        img_emb = self.img_proj(img_emb)
        return F.normalize(img_emb, dim=-1)


def contrastive_loss(img_emb, txt_emb, temperature=0.07):
    logits = img_emb @ txt_emb.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2


def train(cfg):
    icfg = cfg["image_encoder"]
    dcfg = cfg["data"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    item_df = pd.read_parquet(os.path.join(dcfg["processed_dir"], "items.parquet"))
    processor = AutoProcessor.from_pretrained(icfg["model_name"])
    model = AutoModel.from_pretrained(icfg["model_name"]).to(device)

    if icfg["use_lora"]:
        lora_cfg = LoraConfig(
            r=icfg["lora_r"], lora_alpha=icfg["lora_alpha"],
            target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none",
        )
        model = get_peft_model(model, lora_cfg)

    dataset = ImageTitleDataset(item_df, processor)
    loader = DataLoader(
        dataset, batch_size=icfg["batch_size"], shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=icfg["lr"])

    model.train()
    for epoch in range(icfg["epochs"]):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Image encoder epoch {epoch+1}/{icfg['epochs']}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            loss = contrastive_loss(out.image_embeds, out.text_embeds)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} avg loss: {total_loss/len(loader):.4f}")

    # extract final embeddings for ALL items (including ones without a title pair used above)
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for _, row in tqdm(item_df.iterrows(), total=len(item_df), desc="Extracting image embeddings"):
            if pd.isna(row["image_path"]):
                continue
            image = Image.open(row["image_path"]).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            out = model.get_image_features(**inputs) if hasattr(model, "get_image_features") else model(**inputs).image_embeds
            emb = F.normalize(out, dim=-1).cpu().numpy().squeeze()
            embeddings[row["parent_asin"]] = emb

    asins = list(embeddings.keys())
    mat = np.stack([embeddings[a] for a in asins])
    os.makedirs(dcfg["processed_dir"], exist_ok=True)
    np.save(os.path.join(dcfg["processed_dir"], "image_embeddings.npy"), mat)
    with open(os.path.join(dcfg["processed_dir"], "image_embeddings_asins.txt"), "w") as f:
        f.write("\n".join(asins))
    print(f"Saved {len(asins)} image embeddings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
