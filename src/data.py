"""Download a sample of the fashion product catalog and save images + metadata locally."""
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from .config import CATALOG_CSV, DATASET_NAME, IMAGE_DIR, MIN_PER_CLASS, N_PRODUCTS, SEED

META_COLS = ["id", "gender", "masterCategory", "subCategory", "articleType",
             "baseColour", "season", "usage", "productDisplayName"]


def build_catalog(n: int = N_PRODUCTS) -> pd.DataFrame:
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=10_000)

    rows = []
    for ex in tqdm(ds.take(int(n * 1.15)), total=int(n * 1.15), desc="Downloading"):
        if not ex.get("productDisplayName") or ex.get("image") is None:
            continue
        path = IMAGE_DIR / f"{ex['id']}.jpg"
        if not path.exists():
            ex["image"].convert("RGB").save(path, quality=92)
        rows.append({c: ex.get(c) for c in META_COLS} | {"image_path": path.name})

    df = pd.DataFrame(rows).drop_duplicates("id")
    counts = df["articleType"].value_counts()
    df = df[df["articleType"].isin(counts[counts >= MIN_PER_CLASS].index)]
    df = df.head(n).reset_index(drop=True)
    df.to_csv(CATALOG_CSV, index=False)
    print(f"Saved {len(df)} products, {df['articleType'].nunique()} article types -> {CATALOG_CSV}")
    return df


def load_catalog() -> pd.DataFrame:
    if not CATALOG_CSV.exists():
        return build_catalog()
    return pd.read_csv(CATALOG_CSV)


def product_text(row) -> str:
    """Text used for text embeddings: the product title only (category fields would leak the labels)."""
    return str(row["productDisplayName"])


if __name__ == "__main__":
    build_catalog()
