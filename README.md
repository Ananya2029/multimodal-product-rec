# Multimodal Product Recommendation (SigLIP + LLM2Rec-style fusion)

Free-tier feasible implementation of the project plan: fuses a fine-tuned SigLIP
image encoder with a CF-aware text encoder into one item embedding, evaluated on
item-to-item retrieval and (optionally) sequential recommendation.

## 0. Recommended setup (100% free)

1. **Write/edit code in VS Code locally**, push to a **public or private GitHub repo**.
2. **Train on Kaggle Notebooks** (free 30 GPU-hrs/week, T4x2 or P100, stable sessions):
   - New Notebook → Settings → Accelerator → GPU T4 x2 → Internet: On
   - In the first cell:
     ```bash
     !git clone https://github.com/<you>/multimodal-product-rec.git
     %cd multimodal-product-rec
     !pip install -q -r requirements.txt
     ```
   - Run the pipeline (steps below) as notebook cells or `!python src/....py`.
   - Kaggle persists your working directory as notebook "Output" — save embeddings/checkpoints there.
3. **Colab** is fine as a backup, but free-tier GPU availability is inconsistent — don't rely on it for the multi-hour text encoder training step.

## 1. Install locally (for CPU-only dev/debugging, e.g. data_prep dry-runs)

```bash
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 2. Run order

```bash
# 1. Download + filter Amazon Reviews 2023, cache images (~15–30 min depending on category)
python src/data_prep.py --config configs/default.yaml

# 2. Image branch — needs GPU. ~20-40 min for 3 epochs on a Kaggle T4 with 10-15k items.
python src/image_encoder.py --config configs/default.yaml

# 3. Text branch — pick ONE:
#    a) light mode: frozen sentence-transformer + contrastive head. CPU-OK, fast (~10 min).
python src/text_encoder.py --config configs/default.yaml --mode light
#    b) full mode: LLM2Rec-style CSFT+IEM with Qwen2-0.5B+LoRA. Needs GPU, ~1-2 hrs.
python src/text_encoder.py --config configs/default.yaml --mode full

# 4. Fusion — run all 4 modes for the ablation table
python src/fusion.py --config configs/default.yaml --mode image_only
python src/fusion.py --config configs/default.yaml --mode text_only
python src/fusion.py --config configs/default.yaml --mode concat
python src/fusion.py --config configs/default.yaml --mode gated

# 5. Evaluate item-to-item retrieval for each variant
python src/retrieval.py --config configs/default.yaml --embeddings image_embeddings.npy
python src/retrieval.py --config configs/default.yaml --embeddings text_embeddings.npy
python src/retrieval.py --config configs/default.yaml --embeddings fused_concat_embeddings.npy
python src/retrieval.py --config configs/default.yaml --embeddings fused_gated_embeddings.npy
```

## 3. Sequential recommendation (SASRec via RecBole) — optional 2nd downstream task

RecBole expects its own `.inter`/`.item` file format. After step 4 above:
1. Export `data/processed/interactions.parquet` + fused embeddings into RecBole's
   atomic file format (`user_id`, `item_id`, `timestamp` for `.inter`; item embeddings
   as a `.itememb` pretrained-embedding table).
2. Use RecBole's `SASRec` config with `item_pretrained_embedding` pointing to your
   fused embedding matrix, so the fused representation initializes the item embedding
   table instead of training it from scratch.
3. Evaluate with RecBole's built-in Recall@10/20, NDCG@10/20 (same definitions as
   `src/evaluate.py`, provided here too if you want to run it manually on RecBole's
   output rankings).

This step is the most fiddly to wire up — if you're short on time, prioritize the
item-to-item retrieval task (step 5 above), which has a complete pipeline here.

## 4. What's already implemented vs. what's a stub

| Component | Status |
|---|---|
| Data download + 5-core filtering + image caching | ✅ complete (`data_prep.py`) |
| SigLIP image encoder + LoRA contrastive fine-tune | ✅ complete (`image_encoder.py`) |
| Text encoder — light mode (frozen ST + contrastive head) | ✅ complete |
| Text encoder — full mode (CSFT + IEM, Qwen2-0.5B) | ✅ complete, GPU-required |
| Gated / concat fusion + ablation variants | ✅ complete (`fusion.py`) |
| FAISS item-to-item retrieval + Precision@K/NDCG@K | ✅ complete (`retrieval.py`) |
| Modality dropout robustness test | ✅ helper in `evaluate.py` (`modality_dropout_eval`) |
| SASRec/GRU4Rec sequential rec via RecBole | ⚠️ stub — needs RecBole atomic-file export (see §3) |

## 5. Config knobs that matter most for speed vs. quality

In `configs/default.yaml`:
- `data.max_items`: lower to 5000 for a fast first end-to-end pass, raise once the
  pipeline works.
- `text_encoder` mode: use `light` first to validate the whole pipeline cheaply,
  then switch to `full` only once everything else works, since it's the slowest step.
- `image_encoder.epochs` / `text_encoder.csft_epochs`/`iem_epochs`: 1 epoch is enough
  to sanity-check the pipeline runs end-to-end before committing GPU hours to a real run.

## 6. Suggested next steps for you

1. Do a **dry run with `max_items: 500`** locally or on Kaggle CPU to make sure the
   whole pipeline runs start-to-finish without errors — this catches bugs before you
   burn GPU quota.
2. Then scale up to the full `max_items: 15000` on a Kaggle GPU session.
3. Log each ablation run's retrieval metrics into `results/logs/` (a simple CSV: mode, precision@k, ndcg@k) — this becomes your Section 8 ablation table directly.
