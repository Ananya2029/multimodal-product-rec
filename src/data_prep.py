%%writefile src/data_prep.py

"""
data_prep.py
Downloads Amazon Reviews 2023 directly from McAuley Lab,
applies 5-core filtering, builds train/val/test interactions,
and caches product images.

Usage:
    python src/data_prep.py --config configs/default.yaml
"""

import argparse
import os
import json
import gzip
import requests

import yaml
import pandas as pd

from tqdm import tqdm


BASE_URL = (
    "https://datarepo.eng.ucsd.edu/"
    "mcauley_group/data/amazon_2023/raw/"
)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def download_file(url, output_path):
    """Download a file if it does not already exist."""

    if os.path.exists(output_path):
        print(f"Already exists: {output_path}")
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Downloading:")
    print(url)

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f:
        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=os.path.basename(output_path),
        ) as pbar:

            for chunk in response.iter_content(chunk_size=1024 * 1024):

                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    return output_path


def read_jsonl_gz(path):
    """Read gzipped JSONL file into a list of dictionaries."""

    rows = []

    with gzip.open(path, "rt", encoding="utf-8") as f:

        for line in f:

            if line.strip():
                rows.append(json.loads(line))

    return rows


def download_category(category):
    """
    Download Amazon Reviews 2023 review and metadata files
    directly from McAuley Lab.
    """

    print(f"\nLoading reviews for category: {category}")

    review_url = (
        BASE_URL
        + f"review_categories/{category}.jsonl.gz"
    )

    meta_url = (
        BASE_URL
        + f"meta_categories/meta_{category}.jsonl.gz"
    )

    raw_dir = "data/raw/amazon"

    review_path = os.path.join(
        raw_dir,
        f"{category}.jsonl.gz"
    )

    meta_path = os.path.join(
        raw_dir,
        f"meta_{category}.jsonl.gz"
    )

    download_file(review_url, review_path)

    print(f"\nLoading item metadata for category: {category}")

    download_file(meta_url, meta_path)

    print("\nReading review data...")
    reviews = read_jsonl_gz(review_path)

    print("Reading metadata...")
    meta = read_jsonl_gz(meta_path)

    print(f"Reviews loaded: {len(reviews):,}")
    print(f"Metadata loaded: {len(meta):,}")

    return reviews, meta


def five_core_filter(
    df,
    min_interactions=5,
    iterations=5
):
    """
    Iteratively remove users/items with fewer than
    min_interactions interactions.
    """

    for i in range(iterations):

        before = len(df)

        item_counts = df["parent_asin"].value_counts()

        keep_items = item_counts[
            item_counts >= min_interactions
        ].index

        df = df[
            df["parent_asin"].isin(keep_items)
        ]

        user_counts = df["user_id"].value_counts()

        keep_users = user_counts[
            user_counts >= min_interactions
        ].index

        df = df[
            df["user_id"].isin(keep_users)
        ]

        after = len(df)

        print(
            f"5-core iteration {i + 1}: "
            f"{before:,} -> {after:,}"
        )

        if before == after:
            break

    return df.reset_index(drop=True)


def build_item_table(
    meta,
    keep_asins,
    max_items
):

    rows = []

    for row in meta:

        asin = row.get("parent_asin")

        if asin not in keep_asins:
            continue

        images = row.get("images") or {}

        img_url = None

        large_images = images.get("large")

        if large_images:
            img_url = large_images[0]

        description = row.get("description") or []

        if isinstance(description, list):
            description = " ".join(
                str(x) for x in description
            )

        else:
            description = str(description)

        rows.append(
            {
                "parent_asin": asin,
                "title": row.get("title", ""),
                "description": description,
                "category": row.get(
                    "main_category",
                    ""
                ),
                "image_url": img_url,
            }
        )

        if len(rows) >= max_items:
            break

    return pd.DataFrame(rows)


def cache_images(item_df, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    paths = []

    for _, row in tqdm(
        item_df.iterrows(),
        total=len(item_df),
        desc="Downloading images"
    ):

        asin = row["parent_asin"]

        url = row["image_url"]

        out_path = os.path.join(
            out_dir,
            f"{asin}.jpg"
        )

        if os.path.exists(out_path):

            paths.append(out_path)
            continue

        if not url:

            paths.append(None)
            continue

        try:

            response = requests.get(
                url,
                timeout=15
            )

            if response.status_code == 200:

                with open(
                    out_path,
                    "wb"
                ) as f:

                    f.write(response.content)

                paths.append(out_path)

            else:

                paths.append(None)

        except Exception:

            paths.append(None)

    item_df["image_path"] = paths

    return item_df


def build_sequences(df):

    df_sorted = df.sort_values(
        ["user_id", "timestamp"]
    )

    sequences = (
        df_sorted
        .groupby("user_id")["parent_asin"]
        .apply(list)
        .to_dict()
    )

    return sequences


def split_by_user(
    sequences,
    train_ratio,
    val_ratio
):

    train = {}
    val = {}
    test = {}

    for user, seq in sequences.items():

        if len(seq) < 3:

            train[user] = seq

        else:

            train[user] = seq[:-2]
            val[user] = seq[:-1]
            test[user] = seq

    return train, val, test


def main(config_path):

    cfg = load_config(config_path)

    dcfg = cfg["data"]

    category = dcfg["category"]

    reviews, meta = download_category(
        category
    )

    print("\nConverting reviews to DataFrame...")

    reviews_df = pd.DataFrame(reviews)

    required_columns = [
        "user_id",
        "parent_asin",
        "rating",
        "timestamp",
    ]

    missing = [
        c for c in required_columns
        if c not in reviews_df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns in review data: {missing}"
        )

    reviews_df = reviews_df[
        required_columns
    ].dropna()

    print(
        f"Raw interactions: "
        f"{len(reviews_df):,}"
    )

    filtered = five_core_filter(
        reviews_df,
        dcfg["min_interactions"]
    )

    print(
        f"After 5-core filtering: "
        f"{len(filtered):,}"
    )

    keep_asins = set(
        filtered["parent_asin"].unique()
    )

    print(
        f"Unique items after filtering: "
        f"{len(keep_asins):,}"
    )

    item_df = build_item_table(
        meta,
        keep_asins,
        dcfg["max_items"]
    )

    print(
        f"Item metadata table: "
        f"{len(item_df):,} items"
    )

    # Keep only items for which metadata was found
    kept_asins = set(
        item_df["parent_asin"]
    )

    filtered = filtered[
        filtered["parent_asin"].isin(
            kept_asins
        )
    ].copy()

    print(
        f"Interactions after item cap: "
        f"{len(filtered):,}"
    )

    item_df = cache_images(
        item_df,
        dcfg["image_cache_dir"]
    )

    sequences = build_sequences(
        filtered
    )

    train, val, test = split_by_user(
        sequences,
        dcfg["train_ratio"],
        dcfg["val_ratio"]
    )

    processed_dir = dcfg[
        "processed_dir"
    ]

    os.makedirs(
        processed_dir,
        exist_ok=True
    )

    item_df.to_parquet(
        os.path.join(
            processed_dir,
            "items.parquet"
        ),
        index=False
    )

    filtered.to_parquet(
        os.path.join(
            processed_dir,
            "interactions.parquet"
        ),
        index=False
    )

    with open(
        os.path.join(
            processed_dir,
            "sequences_train.json"
        ),
        "w"
    ) as f:

        json.dump(
            train,
            f
        )

    with open(
        os.path.join(
            processed_dir,
            "sequences_val.json"
        ),
        "w"
    ) as f:

        json.dump(
            val,
            f
        )

    with open(
        os.path.join(
            processed_dir,
            "sequences_test.json"
        ),
        "w"
    ) as f:

        json.dump(
            test,
            f
        )

    print(
        "\n========================================"
    )

    print("DATA PREPARATION COMPLETE")

    print(
        "========================================"
    )

    print(
        f"Items: {len(item_df):,}"
    )

    print(
        f"Interactions: {len(filtered):,}"
    )

    print(
        f"Users: "
        f"{filtered['user_id'].nunique():,}"
    )

    print(
        "Images cached: "
        f"{item_df['image_path'].notna().sum():,} "
        f"/ {len(item_df):,}"
    )

    print(
        f"Output directory: {processed_dir}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/default.yaml"
    )

    args = parser.parse_args()

    main(args.config)
