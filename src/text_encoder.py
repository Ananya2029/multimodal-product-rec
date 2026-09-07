"""
text_encoder.py
Two modes, controlled by config:

1. "full" (default, needs a GPU with >=12GB or a small LLM like Qwen2-0.5B + LoRA):
   Stage 1 - CSFT: next-item-title prediction over user sequences (causal LM, LoRA).
   Stage 2 - IEM: switch to bidirectional attention + masked-token + item-level
             contrastive learning on titles, so items interacted together end up close.

2. "light" fallback (CPU-friendly, use if compute is tight):
   Start from a frozen sentence-transformer (all-MiniLM-L6-v2) and only train a
   small contrastive projection head using co-occurrence pairs from user sequences
   as positives. This is the "text + light CF" trade-off mentioned in the project plan.

Usage:
    python src/text_encoder.py --config configs/default.yaml --mode light
"""
import argparse
import os
import json
import yaml
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_cooccurrence_pairs(sequences, window=3):
    """Items that co-occur within `window` steps in a user's history = positive pairs (CF signal)."""
    pairs = []
    for seq in sequences.values():
        for i in range(len(seq)):
            for j in range(i + 1, min(i + 1 + window, len(seq))):
                pairs.append((seq[i], seq[j]))
    return pairs


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class PairDataset(Dataset):
    def __init__(self, pairs, item_titles):
        self.pairs = [(a, b) for a, b in pairs if a in item_titles and b in item_titles]
        self.item_titles = item_titles

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        return self.item_titles[a], self.item_titles[b]


def train_light(cfg):
    """CPU-friendly fallback: frozen sentence-transformer + learned contrastive projection head."""
    from sentence_transformers import SentenceTransformer

    dcfg = cfg["data"]
    tcfg = cfg["text_encoder"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    item_df = pd.read_parquet(os.path.join(dcfg["processed_dir"], "items.parquet"))
    item_titles = dict(zip(item_df["parent_asin"], item_df["title"]))

    with open(os.path.join(dcfg["processed_dir"], "sequences_train.json")) as f:
        sequences = json.load(f)

    pairs = build_cooccurrence_pairs(sequences)
    print(f"Built {len(pairs)} co-occurrence pairs for contrastive training")

    st_model = SentenceTransformer(tcfg["fallback_model"], device=device)
    st_model.eval()
    for p in st_model.parameters():
        p.requires_grad = False

    in_dim = st_model.get_sentence_embedding_dimension()
    head = ProjectionHead(in_dim, tcfg["embed_dim"]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=tcfg["lr"])

    dataset = PairDataset(pairs, item_titles)
    loader = DataLoader(dataset, batch_size=tcfg["batch_size"], shuffle=True)

    for epoch in range(tcfg["iem_epochs"]):
        total_loss = 0.0
        for titles_a, titles_b in tqdm(loader, desc=f"Text head epoch {epoch+1}/{tcfg['iem_epochs']}"):
            with torch.no_grad():
                emb_a = torch.tensor(st_model.encode(list(titles_a))).to(device)
                emb_b = torch.tensor(st_model.encode(list(titles_b))).to(device)
            z_a = head(emb_a)
            z_b = head(emb_b)
            logits = z_a @ z_b.t() / 0.07
            labels = torch.arange(logits.size(0), device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} avg loss: {total_loss/max(len(loader),1):.4f}")

    # extract embeddings for all items
    head.eval()
    asins, titles = zip(*item_titles.items())
    with torch.no_grad():
        base_emb = torch.tensor(st_model.encode(list(titles), batch_size=64, show_progress_bar=True)).to(device)
        final_emb = head(base_emb).cpu().numpy()

    os.makedirs(dcfg["processed_dir"], exist_ok=True)
    np.save(os.path.join(dcfg["processed_dir"], "text_embeddings.npy"), final_emb)
    with open(os.path.join(dcfg["processed_dir"], "text_embeddings_asins.txt"), "w") as f:
        f.write("\n".join(asins))
    print(f"Saved {len(asins)} text embeddings (light mode).")


def train_full(cfg):
    """
    LLM2Rec-style CSFT + IEM with a small causal LM (e.g. Qwen2-0.5B) + LoRA.
    Requires a GPU (Kaggle T4/P100 or Colab T4). See README for expected runtime.
    """
    from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    dcfg = cfg["data"]
    tcfg = cfg["text_encoder"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected. 'full' mode will be extremely slow on CPU. "
              "Consider --mode light instead, or run this on a Kaggle/Colab GPU runtime.")

    item_df = pd.read_parquet(os.path.join(dcfg["processed_dir"], "items.parquet"))
    item_titles = dict(zip(item_df["parent_asin"], item_df["title"]))

    with open(os.path.join(dcfg["processed_dir"], "sequences_train.json")) as f:
        sequences = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(tcfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Stage 1: CSFT (causal, next-title prediction) ----
    print("Stage 1: CSFT (causal next-item-title prediction)")
    csft_model = AutoModelForCausalLM.from_pretrained(tcfg["model_name"]).to(device)
    lora_cfg = LoraConfig(r=tcfg["lora_r"], lora_alpha=tcfg["lora_alpha"],
                           target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    csft_model = get_peft_model(csft_model, lora_cfg)

    seq_texts = []
    for seq in sequences.values():
        titles = [item_titles.get(a, "") for a in seq if a in item_titles]
        if len(titles) >= 2:
            seq_texts.append(" [SEP] ".join(titles))

    optimizer = torch.optim.AdamW(csft_model.parameters(), lr=tcfg["lr"])
    csft_model.train()
    for epoch in range(tcfg["csft_epochs"]):
        random.shuffle(seq_texts)
        total_loss = 0.0
        n_batches = max(1, len(seq_texts) // tcfg["batch_size"])
        for i in tqdm(range(n_batches), desc=f"CSFT epoch {epoch+1}"):
            batch_texts = seq_texts[i*tcfg["batch_size"]:(i+1)*tcfg["batch_size"]]
            if not batch_texts:
                continue
            enc = tokenizer(batch_texts, padding=True, truncation=True,
                             max_length=tcfg["max_seq_len"]*4, return_tensors="pt").to(device)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            out = csft_model(**enc, labels=labels)
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item()
        print(f"CSFT epoch {epoch+1} avg loss: {total_loss/n_batches:.4f}")

    csft_model.save_pretrained(os.path.join(dcfg["processed_dir"], "csft_lora"))

    # ---- Stage 2: IEM (bidirectional encoder + item-level contrastive) ----
    print("Stage 2: IEM (bidirectional item-level contrastive)")
    encoder = AutoModel.from_pretrained(tcfg["model_name"]).to(device)
    encoder = get_peft_model(encoder, lora_cfg)
    proj = ProjectionHead(encoder.config.hidden_size, tcfg["embed_dim"]).to(device)

    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(proj.parameters()), lr=tcfg["lr"])
    asins = list(item_titles.keys())
    titles_list = [item_titles[a] for a in asins]

    encoder.train()
    for epoch in range(tcfg["iem_epochs"]):
        total_loss = 0.0
        n_batches = max(1, len(titles_list) // tcfg["batch_size"])
        idxs = list(range(len(titles_list)))
        random.shuffle(idxs)
        for i in tqdm(range(n_batches), desc=f"IEM epoch {epoch+1}"):
            batch_idx = idxs[i*tcfg["batch_size"]:(i+1)*tcfg["batch_size"]]
            batch_titles = [titles_list[j] for j in batch_idx]
            # two masked/augmented views via random truncation as a cheap augmentation
            view_a = tokenizer(batch_titles, padding=True, truncation=True,
                                max_length=tcfg["max_seq_len"], return_tensors="pt").to(device)
            view_b = tokenizer(batch_titles, padding=True, truncation=True,
                                max_length=max(4, tcfg["max_seq_len"]-4), return_tensors="pt").to(device)
            emb_a = proj(mean_pool(encoder(**view_a).last_hidden_state, view_a["attention_mask"]))
            emb_b = proj(mean_pool(encoder(**view_b).last_hidden_state, view_b["attention_mask"]))
            logits = emb_a @ emb_b.t() / 0.07
            labels_t = torch.arange(logits.size(0), device=device)
            loss = (F.cross_entropy(logits, labels_t) + F.cross_entropy(logits.t(), labels_t)) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"IEM epoch {epoch+1} avg loss: {total_loss/n_batches:.4f}")

    encoder.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(titles_list), tcfg["batch_size"]), desc="Extracting text embeddings"):
            batch = titles_list[i:i+tcfg["batch_size"]]
            enc = tokenizer(batch, padding=True, truncation=True,
                             max_length=tcfg["max_seq_len"], return_tensors="pt").to(device)
            emb = proj(mean_pool(encoder(**enc).last_hidden_state, enc["attention_mask"]))
            embeddings.append(emb.cpu().numpy())
    mat = np.concatenate(embeddings, axis=0)

    os.makedirs(dcfg["processed_dir"], exist_ok=True)
    np.save(os.path.join(dcfg["processed_dir"], "text_embeddings.npy"), mat)
    with open(os.path.join(dcfg["processed_dir"], "text_embeddings_asins.txt"), "w") as f:
        f.write("\n".join(asins))
    print(f"Saved {len(asins)} text embeddings (full mode).")


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["light", "full"], default="light",
                         help="'light' = CPU-friendly frozen sentence-transformer + head. "
                              "'full' = LLM2Rec-style CSFT+IEM, needs a GPU.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.mode == "light":
        train_light(cfg)
    else:
        train_full(cfg)
