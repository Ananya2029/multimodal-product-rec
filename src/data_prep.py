"""
data_prep.py
Downloads Amazon Reviews 2023 (McAuley Lab), applies 5-core filtering,
builds train/val/test interaction splits, and caches product images.

Usage:
    python src/data_prep.py --config configs/default.yaml
"""
import argparse
import os
import json
import yaml
import requests
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def download_category(category):
    """Loads reviews + item metadata for one Amazon Reviews 2023 category via HF datasets hub."""
    print(f"Loading reviews for category: {category}")
    reviews = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{category}",
        trust_remote_code=True,
        split="full",
    )
    print(f"Loading item metadata for category: {category}")
    meta = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_meta_{category}",
        trust_remote_code=True,
        split="full",
    )
    return reviews, meta


def five_core_filter(df, min_interactions=5, iterations=3):
    """Iteratively drop users/items with < min_interactions until stable (standard 5-core)."""
    for _ in range(iterations):
        item_counts = df["parent_asin"].value_counts()
        keep_items = item_counts[item_counts >= min_interactions].index
        df = df[df["parent_asin"].isin(keep_items)]

        user_counts = df["user_id"].value_counts()
        keep_users = user_counts[user_counts >= min_interactions].index
        df = df[df["user_id"].isin(keep_users)]
    return df.reset_index(drop=True)


def build_item_table(meta, keep_asins, max_items):
    rows = []
    for row in meta:
        asin = row.get("parent_asin")
        if asin in keep_asins:
            images = row.get("images", {})
            img_url = None
            if images and images.get("large"):
                large_list = images["large"]
                img_url = large_list[0] if large_list else None
            rows.append({
                "parent_asin": asin,
                "title": row.get("title", ""),
                "description": " ".join(row.get("description", []) or []),
                "category": row.get("main_category", ""),
                "image_url": img_url,
            })
        if len(rows) >= max_items:
            break
    return pd.DataFrame(rows)


def cache_images(item_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for _, row in tqdm(item_df.iterrows(), total=len(item_df), desc="Downloading images"):
        asin = row["parent_asin"]
        url = row["image_url"]
        out_path = os.path.join(out_dir, f"{asin}.jpg")
        if os.path.exists(out_path):
            paths.append(out_path)
            continue
        if not url:
            paths.append(None)
            continue
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(resp.content)
                paths.append(out_path)
            else:
                paths.append(None)
        except Exception:
            paths.append(None)
    item_df["image_path"] = paths
    return item_df


def build_sequences(df):
    """Group interactions by user, sorted by timestamp, for sequential rec + text CSFT."""
    df_sorted = df.sort_values(["user_id", "timestamp"])
    sequences = df_sorted.groupby("user_id")["parent_asin"].apply(list).to_dict()
    return sequences


def split_by_user(sequences, train_ratio, val_ratio):
    """Leave-one-out style split per user (last item = test, second-last = val)."""
    train, val, test = {}, {}, {}
    for user, seq in sequences.items():
        if len(seq) < 3:
            train[user] = seq
            continue
        train[user] = seq[:-2]
        val[user] = seq[:-1]
        test[user] = seq
    return train, val, test


def main(config_path):
    cfg = load_config(config_path)
    dcfg = cfg["data"]

    reviews, meta = download_category(dcfg["category"])

    print("Converting reviews to DataFrame...")
    reviews_df = pd.DataFrame(reviews)
    reviews_df = reviews_df[["user_id", "parent_asin", "rating", "timestamp"]].dropna()

    print(f"Raw interactions: {len(reviews_df)}")
    filtered = five_core_filter(reviews_df, dcfg["min_interactions"])
    print(f"After 5-core filtering: {len(filtered)}")

    keep_asins = set(filtered["parent_asin"].unique())
    print(f"Unique items after filtering: {len(keep_asins)}")

    item_df = build_item_table(meta, keep_asins, dcfg["max_items"])
    print(f"Item metadata table: {len(item_df)} items (capped at max_items)")

    # re-filter interactions to the capped item set
    kept_asins_capped = set(item_df["parent_asin"])
    filtered = filtered[filtered["parent_asin"].isin(kept_asins_capped)]

    item_df = cache_images(item_df, dcfg["image_cache_dir"])

    sequences = build_sequences(filtered)
    train, val, test = split_by_user(sequences, dcfg["train_ratio"], dcfg["val_ratio"])

    os.makedirs(dcfg["processed_dir"], exist_ok=True)
    item_df.to_parquet(os.path.join(dcfg["processed_dir"], "items.parquet"))
    filtered.to_parquet(os.path.join(dcfg["processed_dir"], "interactions.parquet"))
    with open(os.path.join(dcfg["processed_dir"], "sequences_train.json"), "w") as f:
        json.dump(train, f)
    with open(os.path.join(dcfg["processed_dir"], "sequences_val.json"), "w") as f:
        json.dump(val, f)
    with open(os.path.join(dcfg["processed_dir"], "sequences_test.json"), "w") as f:
        json.dump(test, f)

    print("Done. Files written to", dcfg["processed_dir"])
    print(f"Items with a cached image: {item_df['image_path'].notna().sum()} / {len(item_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
