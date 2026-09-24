# Demo guide: Multimodal Product Recommendation

## Before the demo (5 minutes)
1. **Free up memory.** Restart the PC, or close Docker/WSL, Oracle, MySQL, WhatsApp and extra browser tabs. The AI models need about 1–2 GB of free RAM.
2. Open PowerShell and start the app:
   ```
   cd "D:\D drive\Project\claude\multimodal_recsys"
   python -m streamlit run app.py
   ```
3. The browser opens at **http://localhost:8501**. **Warm it up once:** go to *💬 Text search* and wait about 40 seconds while CLIP loads. After that every search is quick.
4. Keep one product photo ready on the desktop, e.g. any shoe or watch from `data\images\`, for the image-search step.

## Demo flow (about 10 minutes)

| # | Section | What to do | What to say |
|---|---|---|---|
| 1 | 📊 Model comparison | Show the green winner banner and the 3 winner cards | "We compared 11 systems: text-only, image-only and multimodal. **CLIP image + text is the best overall.** The wins are statistically tested: 95% confidence intervals and a paired bootstrap test." |
| 2 | 📊 (scroll) | The chart with error bars, then the fusion-weight sweep | "Combining image and text beats either alone. The best mix is 30–60% image." |
| 3 | 📊 (scroll) | 🧠 **SGD vs Adam vs AdamW** table and curves | "We trained a fusion network with 3 optimizers, each tuned on validation data and run with 3 seeds. Training lifts accuracy from **0.76 to 0.92** on unseen products. SGD is slightly best on text search; Adam and AdamW are almost identical." |
| 4 | 📊 (scroll) | 🎨 GAN metrics | "A conditional GAN generates new product designs. Its realism improved 30% over an untrained generator, but it's still far from real photos. That's an honest limitation of a small CPU-trained GAN." |
| 5 | 🧥 More like this | Press 🎲 **Random product** 2–3 times | "Pick any product and get similar ones; ✅ means the same product type." |
| 6 | 💬 Text search | Type `red dress for women`, then `black analog watch` | "Free-text search. CLIP matches words directly against the product *photos*." |
| 7 | Sidebar | Switch the model to **TF-IDF (title)**, then back to **CLIP image + text** | "Here you can see the difference between models live." |
| 8 | 📷 Image search | Upload the prepared photo | "Search by photo." |
| 9 | 🧩 Image + text | Same photo + `in red colour` | "Composed query: *this product, but red*." |
| 10 | 🌐 Internet | Paste an image link from a shopping site; search the live shop for `sunglasses` | "It works with images from the internet, and with a live product API (dummyjson.com, 194 products)." |
| 11 | 🎨 GAN | Category **Watches** or **Topwear**, change the seed | "The GAN designs a new product, and the recommender finds similar real ones." |
| 12 | (optional) REST API | Second terminal: `python -m uvicorn api:app --port 8000`, open http://localhost:8000/docs | "The recommender is also a REST API that other apps can call." |

## Key numbers

| Result | Value |
|---|---|
| Catalog | 3,000 products, 66 types, from Hugging Face `ashraq/fashion-product-images-small` |
| Best zero-shot system | **CLIP image + text**, mean rank 1.33 of 11 |
| "More like this" (same type), best | DINOv2 + MiniLM **0.835** NDCG@10 (CLIP image + text 0.815) |
| Same type + colour, best | CLIP image + text **0.527** (p < 0.001) |
| Text search, best | CLIP image + text **0.625** (statistically tied with SigLIP image, 0.618) |
| Trained fusion head (900 unseen products) | 0.763 → **0.920** (SGD 0.920 ± 0.003, Adam 0.919, AdamW 0.919) |
| GAN (50 epochs, 31 min on CPU) | FD-CLIP 0.50 (untrained 0.72, real 0.035); category accuracy 14% (chance 7%) |
| Tests | Metrics, all 14 systems in every app section, trained heads, GAN, URL safety, REST API |

## Likely questions

**Which optimizer did you use?**
The pretrained encoders were not retrained. For the trained fusion network we compared **SGD** (momentum 0.9, lr 0.05), **Adam** (lr 3e-3) and **AdamW** (lr 3e-3, weight decay 0.05). Each was tuned on a validation split, with 3 seeds. The GAN uses **Adam with TTUR** (discriminator lr 4e-4, generator lr 1e-4).

**Why is the optimizer difference so small?**
The network is small, the loss is smooth, and all three were tuned. What matters most is that we train at all: +0.16 NDCG. Adam ≈ AdamW because weight decay barely changes the weights in 40 short epochs.

**Why is CLIP the best?**
It was trained on 400 million image–caption pairs, so images and text share one embedding space. That lets it combine visual similarity with colour and type words from the titles.

**How do you know the winner is really better?**
Bootstrap 95% confidence intervals, and a paired bootstrap significance test against the runner-up. We also report honestly where there's no significant difference (text search).

**Why use a GAN in a recommender?**
It's a "design a new product" feature: generate a concept, then recommend real items like it. On its own a GAN doesn't improve recommendations, because recommendation is a retrieval task.

**What problems did you face?**
- The first GAN run **mode-collapsed** (identical images). We fixed it with DiffAugment, a minibatch-std feature and TTUR.
- Brand names in titles bias text similarity toward the same brand.
- Low RAM caused crashes. We fixed it with lazy loading of each model half and a free-memory guard.

**Limitations and future work**
Relevance labels are metadata proxies with no real user clicks. Future work: a larger catalog on a GPU, fine-tuning CLIP, removing brand names from titles, and a longer GAN run (e.g. on Google Colab: `python -m src.gan --epochs 300`).

## If something goes wrong during the demo
| Problem | Fix |
|---|---|
| "Not enough free memory to load SigLIP" | Expected on a low-RAM PC. Use **CLIP image + text** (the default). |
| Page blank for a while | The first model load takes about 40 s. Wait. |
| Internet URL fails | Use a *direct* image link (ends in .jpg/.png), or skip that step. |
| App stopped | In PowerShell, run `python -m streamlit run app.py` again. |
