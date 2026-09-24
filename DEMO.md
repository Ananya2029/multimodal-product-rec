# Demo guide: Multimodal Product Recommendation (trained from scratch)

## Before the demo (5 minutes)
1. Close heavy programs: VS Code, Docker, WhatsApp, extra browser tabs.
2. Start the app from PowerShell:
   ```
   cd "D:\D drive\Project\claude\multimodal_recsys"
   python -m streamlit run app.py
   ```
3. The browser opens at **http://localhost:8501**. If the app was already running, stop it with Ctrl+C and start it again so it loads the latest version.
4. Keep one product photo on the desktop for image search, e.g. any file from `data\scratch\test_images\`.

## What the project is (30-second pitch)
"We recommend products by learning **one embedding from the product photo and the title together**. Everything is trained **from scratch**: a CNN for images, a Transformer for titles, and our own tokenizer, with no pretrained weights. We compare 6 ways of combining the two modalities, 3 optimizers and 5 training components, on **44,013 products**. Each result is tested on **6,602 products the model never saw**, with 3 random seeds and significance tests."

## Demo flow (about 10 minutes)
The app has two pages: **📊 Results** and **🛍️ Try the proposed model**.

| # | Page | What to do | What to say |
|---|---|---|---|
| 1 | 📊 Results: *What we did* | Walk through steps 1–5 and the architecture diagram | "Photo → CNN, title → Transformer, both trained from scratch, combined by our gated fusion into one product embedding." |
| 2 | 📊 *What we achieved* | The 4 green metric cards | "Our model on 6,602 unseen products: 0.956 for similar products, +0.14 over image-only; better photo search than an image-only model." |
| 3 | (scroll) | Key findings, then the comparison table (our model is highlighted) | "Compared with 5 baselines over 3 seeds, with significance tests." |
| 4 | (scroll) | Optimizer table and curves | "SGD, Adam and AdamW, each tuned on validation; SGD selected." |
| 5 | (scroll) | Ablation table | "The supervised contrastive loss matters most." |
| 6 | (scroll) | GAN | "A conditional GAN, also from scratch, generates new product images." |
| 7 | 🛍️ Try it: 🧥 Similar to a product | 🎲 Random product, 2–3 times | "Products the model never saw in training; ✅ = same type." |
| 8 | 💬 Text | `black handbag for women`, `blue jeans for men` | "Search by words." |
| 9 | 📷 Photo | Upload the prepared photo | "Search by photo." |
| 10 | 🧩 Photo + text | Photo + `red` | "Both together in one query." |

## Results (test NDCG@10, 6,602 unseen products, mean of 3 seeds)
| Method | More like this (type) | Type + colour | Text query | Photo query |
|---|---|---|---|---|
| Image-only CNN | 0.813 | 0.372 | n/a | 0.813 |
| Text-only Transformer | **0.956** | **0.779** | 0.548 | n/a |
| Early fusion | 0.955 | 0.752 | 0.552 | 0.828 |
| Late fusion | 0.954 | 0.735 | **0.680** | 0.811 |
| **Gated fusion (proposed)** | **0.956** | 0.765 | 0.568 | 0.831 |
| **Cross-attention (proposed)** | **0.956** | 0.761 | 0.561 | **0.833** |

Optimizers (gated): SGD 0.956 / 0.765 / 0.568 / **0.831**; Adam 0.956 / 0.752 / 0.563 / 0.823; AdamW **0.957** / 0.757 / **0.585** / 0.821.

## Key findings (say these honestly)
1. **Adding text to images helps a lot.** Image-only reaches 0.813 on "more like this"; every model that uses the title reaches about 0.956 (+0.14).
2. **The proposed fusion methods are best for photo queries.** Cross-attention scores 0.833 and gated 0.831, against 0.813 for image-only; the gain is statistically significant (p < 0.001). The title information learned during training improves search by photo.
3. **The gated model is best or tied-best on "more like this"**, and significantly better than early and late fusion (p < 0.02).
4. **Different fusion methods win different tasks.** Late fusion is best for text queries (0.680). Text-only is best on type + colour, because product titles literally contain the type and colour words ("Men Navy Blue Shirt").
5. **Ablations:**
   - The supervised contrastive loss is essential (−0.17 without it).
   - The image–text contrastive loss helps text queries (−0.08 without it).
   - Modality dropout and the consistency loss did **not** help. We report that honestly.

## Likely questions
- **Why doesn't the proposed model win everything?** Titles in this catalog contain the answer words (type, colour), so text alone is very strong. Fusion helps where text is missing, i.e. photo queries. Next step: test with noisy or missing titles, which is where image features matter most.
- **Why train from scratch?** To study the fusion methods themselves, without the advantage of large-scale pretraining. As a reference only, pretrained CLIP was also evaluated earlier (results in `results/model_comparison.csv`); it is not used in our method or the app.
- **Which optimizer?** SGD (Nesterov, lr 0.1) was selected on validation. The three optimizers are within about 0.01 of each other; SGD is best on photo and type + colour, and AdamW on text queries.
- **How long did training take?** About 50 training runs (about 6 minutes each) on a Kaggle T4 GPU, roughly 4–5 hours in total.
- **Is it reproducible?** Yes. Fixed seeds, a notebook (`colab/`), all runs logged in `results/scratch/all_runs.csv`, and tests (`python run_tests.py`).
