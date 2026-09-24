import os
from pathlib import Path

# Offline mode: if every model is already downloaded, don't contact huggingface.co at all (faster start-up,
# works without internet). Must run before transformers / huggingface_hub / datasets are imported.
MODEL_IDS = ["openai/clip-vit-base-patch32", "google/siglip-base-patch16-224", "facebook/dinov2-small",
             "sentence-transformers/all-MiniLM-L6-v2"]
_hf_cache = Path(os.environ.get("HF_HUB_CACHE")
                 or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
if all((_hf_cache / ("models--" + m.replace("/", "--"))).exists() for m in MODEL_IDS):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
CATALOG_CSV = DATA_DIR / "catalog.csv"
EMB_DIR = ROOT / "embeddings"
RESULTS_DIR = ROOT / "results"

DATASET_NAME = "ashraq/fashion-product-images-small"
N_PRODUCTS = 3000          # catalog size (CPU-friendly); raise if you have a GPU
MIN_PER_CLASS = 5          # drop article types with fewer items (can't evaluate them)
SEED = 42
BATCH_SIZE = 32

# Fusion weight for image vs. text when combining modalities (0 = text only, 1 = image only)
DEFAULT_IMAGE_WEIGHT = 0.5

for d in (DATA_DIR, IMAGE_DIR, EMB_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
