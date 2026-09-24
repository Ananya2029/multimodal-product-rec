# Multimodal Product Recommendation with Image and Text Embeddings Learned from Scratch

We recommend products by learning a **joint embedding of product photos and titles**. Every encoder is trained **from random initialisation**: no ImageNet, CLIP or BERT weights, and no pretrained tokenizer. The study compares image-only, text-only and four fusion methods, including two proposed ones. It also compares three optimizers (SGD, Adam, AdamW) and runs ablations of each training component.

## Method (`src/scratch/`)

| Component | Design |
|---|---|
| Image encoder | ResNet-style CNN (6 residual blocks, 64×64 input → 8×8×256 regions → 256-d) |
| Text encoder | Own tokenizer (vocabulary from *training* titles only) → word embeddings + positions → 2-layer Transformer → 256-d |
| Fusion methods | **image**, **text**, **early** (concatenation + MLP), **late** (weighted score sum), **gated** (proposed: per-dimension learned gate between modalities), **xattn** (proposed: title tokens attend over image regions) |
| Missing modality | learned null vectors + **modality dropout** in training, so every fused model also answers text-only or photo-only queries |
| Loss | supervised contrastive loss on product type (+ type & colour) + symmetric image–text contrastive loss (InfoNCE) + **missing-modality consistency loss** (proposed: the text-only and image-only fused embeddings are pulled toward the full image + text embedding, with stop-gradient) |
| Optimizers | SGD (Nesterov momentum), Adam, AdamW, each with its learning rate tuned on validation; cosine schedule with warm-up |

## Experimental protocol (`src/scratch/experiments.py`)
1. **Data:** the full *Fashion Product Images (small)* catalog (about 44k products). Product types with fewer than 20 items are dropped. Stratified **70/15/15** train/val/test split. The **test products are never used** for training, tuning or model selection.
2. **Learning-rate tuning** for each optimizer on the validation split.
3. **Optimizer comparison:** proposed gated model × {SGD, Adam, AdamW} × 3 seeds. The best optimizer is chosen **on validation**.
4. **Method comparison:** 6 methods × best optimizer × 3 seeds.
5. **Ablations:** the gated model without the image–text loss, without the supervised loss, without the consistency loss, and without modality dropout.
6. **Metrics:** NDCG@10 on the test products for four tasks: *more like this* (same type; same type + colour), *text query → product* and *photo query → product*. Results are reported as mean ± std over seeds, with **paired bootstrap significance tests** of each proposed method against every baseline.

## How to run

**1. Train on Google Colab (GPU, about 3–4 hours, resumable)**
1. `python colab/make_code_zip.py` creates `colab/mmrec_code.zip`.
2. Open `colab/train_from_scratch.ipynb` in Colab, select a T4 GPU, and choose **Run all**. When asked, upload `mmrec_code.zip`. Progress is saved to Google Drive.
3. Unzip the downloaded `mmrec_scratch_results.zip` into this folder.

**2. Local app (CPU; the models are small, 0.4–4M parameters)**
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```
The app offers: 📊 **Model & results** (method, optimizer and ablation tables, significance tests, GAN results, pretrained reference), 🧥 more like this, 💬 text search, 📷 image search and 🧩 image + text. Its catalog is the unseen test split.

**Tests:** `python run_tests.py`. **Local end-to-end check:** `python -m src.scratch.data --source local`, then `python -m src.scratch.experiments --smoke`.

## Results
The from-scratch results (tables and figures) are written to `results/scratch/` by the Colab run: `method_comparison.csv/.png`, `optimizer_comparison.csv/.png`, `optimizer_curves.png`, `ablations.csv/.png`, `significance.csv`, `summary.json`.

### Conditional GAN (`src/gan.py`, trained from scratch)
A conditional DCGAN (hinge loss, spectral norm, minibatch-std, DiffAugment, Adam with TTUR) generates 64×64 product images for each category. The first run mode-collapsed; DiffAugment fixed it. After 50 epochs on CPU, the FD in CLIP space fell from 0.72 (untrained generator) to **0.50** (real images score 0.035). Category accuracy is **14%** (chance is 7%).

### Reference only: pretrained encoders (`src/evaluate.py`)
To show how far large-scale pretraining reaches, the same tasks were scored on a 3,000-product sample with **pretrained** encoders (CLIP, SigLIP, DINOv2, MiniLM, ResNet-50) and TF-IDF. These models are **not** part of our method. The best was CLIP image + text (NDCG@10: 0.815 same type, 0.527 same type + colour, 0.625 text search). See `results/model_comparison.csv` and `results/summary.json`.

## Project layout
```
src/scratch/data.py         44k catalog -> 64x64 images, tokenizer, stratified split
src/scratch/models.py       CNN, Transformer, 6 fusion methods (gated & cross-attention proposed)
src/scratch/train.py        losses, training loop, early stopping, test metrics
src/scratch/experiments.py  tuning, optimizer / method comparison, ablations, significance, figures, export
src/scratch/serve.py        loads exported models for the app
colab/                      Colab notebook + code packager
app.py                      Streamlit demo (from-scratch models only)
src/gan.py                  conditional DCGAN
src/evaluate.py, ...        pretrained reference baselines, REST API (api.py), earlier experiments
tests/                      pytest suite (python run_tests.py)
```
