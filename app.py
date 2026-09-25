"""Multimodal product recommendation with image and text embeddings learned from scratch - Streamlit app.

    python -m streamlit run app.py

Two pages: the study (our approach, what we achieved) and a demo of the proposed gated-fusion model.
The demo catalog is the TEST split: 6,602 products the model never saw during training.
"""
import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.config import RESULTS_DIR
from src.scratch import serve
from src.scratch.models import METHOD_NAMES

st.set_page_config(page_title="Multimodal Product Recommendation", page_icon="🛍️", layout="wide")
PROPOSED = "gated"
RES = serve.RES_DIR
METRICS = {"i2i": "Similar products (same type)", "i2i_strict": "Similar products (same type + colour)",
           "text2item": "Search by text", "image2item": "Search by photo"}

st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
div[data-testid="stMetric"] {background: rgba(46,139,87,0.08); border-radius: 10px; padding: 12px 16px;}
.step {border-left: 4px solid #2E8B57; padding: 4px 0 4px 14px; margin-bottom: 10px;}
.card-title {font-size: 0.85rem; font-weight: 600; line-height: 1.2; height: 2.4em; overflow: hidden;}
.card-sub {font-size: 0.75rem; opacity: 0.75;}
</style>""", unsafe_allow_html=True)


def read_csv(name):
    f = RES / name
    return pd.read_csv(f) if f.exists() else None


def fmt(mean, std):
    return "–" if pd.isna(mean) else f"{mean:.3f} ± {std:.3f}"


# ------------------------------------------------------------------ header + navigation
st.title("🛍️ Multimodal Product Recommendation")
st.caption("Image and text embeddings fused with a **gated fusion** network")
page = st.radio("Page", ["📊 Results", "🛍️ Try the proposed model"], horizontal=True,
                label_visibility="collapsed", key="page")

summary_f = RES / "summary.json"
if not (summary_f.exists() and serve.available()):
    st.warning("Results not found. Unzip `mmrec_scratch_results.zip` (from the Colab/Kaggle notebook) into the "
               "project folder.")
    st.stop()
summ = json.loads(summary_f.read_text())
meth = read_csv("method_comparison.csv").set_index("method")
opt = read_csv("optimizer_comparison.csv").set_index("optimizer")
abl = read_csv("ablations.csv").set_index("ablation")
sig = read_csv("significance.csv")
cat = serve.catalog()
n_types = pd.read_csv(serve.SCRATCH_DIR / "meta.csv", usecols=["articleType"])["articleType"].nunique()

# ================================================================== RESULTS
if page == "📊 Results":
    # ------------------------------------------------------------ our approach
    st.header("Our Approach")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown(f"""
<div class="step"><b>1 · Data.</b> {summ['n_products']:,} fashion products (photo + title) in
{n_types} product types, split 70 / 15 / 15 into train / validation / test.
The <b>{summ['n_test']:,} test products are never seen</b> during training or tuning.</div>
<div class="step"><b>2 · Encoders, trained from scratch.</b> A ResNet-style <b>CNN</b> reads the 64×64 photo; our own
tokenizer and a <b>Transformer</b> read the title.</div>
<div class="step"><b>3 · Proposed gated fusion.</b> A learned gate decides, for every feature and every product,
how much to trust the image vs. the title, and produces one 128-d product embedding.
Products with close embeddings are recommended.</div>
<div class="step"><b>4 · Training.</b> Supervised contrastive loss (same product type ⇒ close) + image–text
contrastive loss (photo and title of a product ⇒ close). Optimizers <b>SGD, Adam and AdamW</b> were each tuned
on validation; {summ['epochs']} epochs; {len(summ['seeds'])} random seeds; Kaggle T4 GPU.</div>
<div class="step"><b>5 · Evaluation.</b> NDCG@10 on the unseen test products for four tasks: similar products
(same type / same type + colour), search by text, search by photo. We compare against 5 baselines and
test significance with a paired bootstrap.</div>
""", unsafe_allow_html=True)
    with c2:
        st.graphviz_chart("""
digraph {
  rankdir=TB; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
  photo [label="Product photo\\n64×64", fillcolor="#dbe9f6"];
  title [label="Product title\\n\\"Men Navy Blue Shirt\\"", fillcolor="#dbe9f6"];
  cnn [label="CNN\\n(6 residual blocks)", fillcolor="#fde2c8"];
  tr [label="Transformer\\n(own tokenizer)", fillcolor="#fde2c8"];
  gate [label="Gated fusion\\ng = σ(W[h_img ; h_txt])\\nh = g·h_img + (1−g)·h_txt", fillcolor="#c9ecd6"];
  emb [label="Product embedding\\n(128-d)", fillcolor="#c9ecd6"];
  rec [label="Recommendations\\n(nearest products)", fillcolor="#eeeeee"];
  photo -> cnn -> gate; title -> tr -> gate; gate -> emb -> rec;
}""")

    # ------------------------------------------------------------ what we achieved
    st.header("What we achieved")
    g, img = meth.loc[PROPOSED], meth.loc["image"]
    cols = st.columns(4)
    for col, (k, label) in zip(cols, METRICS.items()):
        base = img.get(f"{k} mean") if k != "text2item" else meth.loc["text"].get(f"{k} mean")
        base_name = "image-only" if k != "text2item" else "text-only"
        col.metric(label, f"{g[f'{k} mean']:.3f}",
                   None if pd.isna(base) else f"{g[f'{k} mean'] - base:+.3f} vs {base_name}")
    st.caption(f"NDCG@10 of the proposed gated-fusion model on {summ['n_test']:,} unseen test products "
               f"(mean of {len(summ['seeds'])} seeds; 1.0 = perfect ranking).")

    def p_of(metric, other):
        r = sig[(sig["metric"] == metric) & (sig["proposed"] == PROPOSED) & (sig["vs"] == other)]
        return None if r.empty else float(r["p_value"].iloc[0])

    p_img = p_of("image2item", "image")
    st.markdown(f"""
**Key findings**
- **Combining photo and title works far better than the photo alone:** similar-product accuracy rises from
  {img['i2i mean']:.3f} to **{g['i2i mean']:.3f}**, and same type + colour from {img['i2i_strict mean']:.3f}
  to **{g['i2i_strict mean']:.3f}**.
- **The fused model finds products from a photo better than an image-only model:**
  {g['image2item mean']:.3f} vs {img['image2item mean']:.3f} (p {'< 0.001' if p_img is not None and p_img < 0.001 else f'= {p_img:.3f}'}):
  what it learned from titles improves search by photo.
- **One model answers every query type:** text, photo, or photo + text, via a shared embedding space.
- **Gated fusion is the best or tied-best fusion for similar products**, and significantly better than early
  and late fusion.
- **Optimizer:** SGD, Adam and AdamW are within about 0.01 of each other; **{summ['best_optimizer']}** was selected
  on validation.
""")

    # ---- comparison table + chart
    st.subheader("Comparison with baselines")
    order = ["image", "text", "early", "late", "gated", "xattn"]
    table = pd.DataFrame({"Model": [METHOD_NAMES[m] for m in order if m in meth.index]})
    for k, label in METRICS.items():
        table[label] = [fmt(meth.loc[m, f"{k} mean"], meth.loc[m, f"{k} std"]) for m in order if m in meth.index]
    table["Parameters"] = [f"{meth.loc[m, 'params (M)']:.1f} M" for m in order if m in meth.index]

    def highlight(row):
        return ["background-color: rgba(46,139,87,0.18); font-weight: 600" if "Gated" in row["Model"] else ""
                for _ in row]
    st.dataframe(table.style.apply(highlight, axis=1), width="stretch", hide_index=True)
    st.caption("NDCG@10, mean ± std over 3 seeds, unseen test products. Text-only is strong on "
               "*same type + colour* because product titles literally contain the type and colour words.")
    if (RES / "method_comparison.png").exists():
        st.image(str(RES / "method_comparison.png"), width="stretch")

    # ---- optimizers
    st.subheader("Optimizer comparison: SGD vs Adam vs AdamW")
    ot = pd.DataFrame({"Optimizer": opt.index, "Learning rate (tuned)": opt["lr"].values})
    for k, label in METRICS.items():
        ot[label] = [fmt(opt.loc[o, f"{k} mean"], opt.loc[o, f"{k} std"]) for o in opt.index]
    ot["Training time"] = [f"{t:.0f} min" for t in opt["train min"]]
    st.dataframe(ot, width="stretch", hide_index=True)
    if (RES / "optimizer_curves.png").exists():
        st.image(str(RES / "optimizer_curves.png"), width="stretch")

    # ---- ablations
    st.subheader("What each part of the model contributes (ablation)")
    order_a = ["full", "no_supcon", "no_itc", "no_consistency", "no_modality_dropout"]
    at = pd.DataFrame({"Variant": [abl.loc[a, "name"] for a in order_a if a in abl.index]})
    for k, label in METRICS.items():
        at[label] = [f"{abl.loc[a, f'{k} mean']:.3f}" for a in order_a if a in abl.index]
    st.dataframe(at, width="stretch", hide_index=True)
    st.caption("Removing the supervised contrastive loss costs the most; removing the image–text contrastive loss "
               "hurts search by text and photo. Modality dropout and the consistency loss did not help here.")

    # ---- robustness to damaged titles
    rob = read_csv("robustness_summary.csv")
    if rob is not None:
        st.subheader("Robustness: what if product titles are missing or wrong?")
        st.markdown("Real catalogs have short, missing or wrong titles. We damaged the titles of the unseen test "
                    "products (photos untouched) and re-scored each model: **similar products (same type)**, NDCG@10, "
                    "mean of 3 random draws.")
        conds = ["clean", "drop 30%", "drop 60%", "no title", "wrong 20%", "wrong 50%"]
        labels = {"clean": "Clean titles", "drop 30%": "30% of words removed", "drop 60%": "60% removed",
                  "no title": "No title", "wrong 20%": "20% wrong titles", "wrong 50%": "50% wrong titles"}
        piv = rob.pivot(index="model", columns="condition", values="i2i mean").reindex(columns=conds)
        piv = piv.reindex([m for m in ["image", "text", "early", "late", "gated", "xattn"] if m in piv.index])
        rt = pd.DataFrame({"Model": [METHOD_NAMES[m] for m in piv.index]})
        for c in conds:
            rt[labels[c]] = [f"{v:.3f}" for v in piv[c]]
        st.dataframe(rt.style.apply(highlight, axis=1), width="stretch", hide_index=True)
        gt, tt, it = piv.loc["gated"], piv.loc["text"], piv.loc["image"]
        st.markdown(f"""
- **Fusion keeps working when text fails:** with no titles, the text-only model collapses to {tt['no title']:.3f},
  while gated fusion keeps **{gt['no title']:.3f}**, because the image branch carries the recommendation.
- **Open problem:** with damaged titles, the fused models fall below the image-only model
  ({gt['drop 60%']:.3f} vs {it['drop 60%']:.3f} at 60% words removed). They trust unreliable text too much, which
  motivates our next step: noise-aware training.
""")
        if (RES / "robustness.png").exists():
            st.image(str(RES / "robustness.png"), width="stretch")

    # ---- GAN
    gan_json = RESULTS_DIR / "gan_metrics.json"
    if gan_json.exists():
        gm = json.loads(gan_json.read_text())
        st.subheader("Generating new products with a GAN")
        a, b = st.columns([1, 1.3])
        with a:
            st.markdown("A **conditional DCGAN**, also trained from scratch, generates new 64×64 product images "
                        "for a chosen category (DiffAugment fixed an initial mode collapse).")
            st.metric("Realism vs untrained generator (FD, lower is better)",
                      f"{gm['FD-CLIP (generated vs real)']:.2f}",
                      f"{gm['FD-CLIP (generated vs real)'] - gm['FD-CLIP untrained generator (baseline)']:+.2f}",
                      delta_color="inverse")
            st.metric("Recognised as the requested category", f"{gm['class accuracy (generated)']:.0%}",
                      f"chance {gm['chance accuracy']:.0%}", delta_color="off")
        if (RESULTS_DIR / "gan_samples.png").exists():
            b.image(str(RESULTS_DIR / "gan_samples.png"), caption="Generated samples per category", width="stretch")

# ================================================================== DEMO
else:
    st.markdown(f"**Model: Gated fusion (proposed)** · catalog of **{len(cat):,} products it never saw in "
                "training**")
    mode = st.radio("Search by", ["💬 Text", "📷 Photo", "🧩 Photo + text", "🧥 Similar to a product"],
                    horizontal=True, key="mode")
    k = st.slider("Number of results", 5, 20, 10, 5, key="k")

    def results(q, exclude=None, ref_type=None):
        idx, sc = serve.top_k(serve.scores(PROPOSED, q), k, exclude)
        hits = [(i, v) for i, v in zip(idx, sc) if np.isfinite(v)]
        cols = st.columns(5)
        for n, (i, v) in enumerate(hits):
            r = cat.iloc[i]
            mark = "" if ref_type is None else (" ✅" if r["articleType"] == ref_type else "")
            with cols[n % 5]:
                st.image(r["image_file"], width="stretch")
                st.markdown(f"<div class='card-title'>{r['productDisplayName']}</div>"
                            f"<div class='card-sub'>{r['articleType']} · {r['baseColour']} · match {v:.2f}{mark}"
                            f"</div>", unsafe_allow_html=True)

    def upload(label, key):
        up = st.file_uploader(label, type=["jpg", "jpeg", "png", "webp"], key=key)
        if not up:
            return None
        try:
            return Image.open(up).convert("RGB")
        except Exception:
            st.error("That file isn't a readable image. Please upload a JPG, PNG or WEBP photo.")
            return None

    if mode == "💬 Text":
        text = st.text_input("What are you looking for?", "black handbag for women", key="q_text")
        if text.strip():
            results(serve.encode_query(PROPOSED, text=text))
    elif mode == "📷 Photo":
        img = upload("Upload a product photo", "q_img")
        if img is not None:
            st.image(img, width=160)
            results(serve.encode_query(PROPOSED, image=img))
        else:
            st.info("Upload a photo of a shoe, watch, bag, t-shirt, ...")
    elif mode == "🧩 Photo + text":
        a, b = st.columns([1, 2])
        with a:
            img = upload("Reference photo", "q_combo")
        text = b.text_input("Plus words: a colour and the product type", "", key="q_combo_text",
                            placeholder="e.g. white watch, black tshirt, white sports shoes")
        b.caption("Tip: write the colour together with the product type (\"white watch\"). A colour word alone "
                  "can outweigh the photo.")
        if img is not None:
            a.image(img, width=140)
            results(serve.encode_query(PROPOSED, image=img, text=text))
    else:
        # the button writes the dropdown's own state *before* the dropdown is drawn (a keyed widget ignores
        # later changes to its `index` argument, which is why the button previously had no effect)
        if "q_item" not in st.session_state:
            st.session_state.q_item = int(np.random.default_rng(0).integers(len(cat)))

        def pick_random():
            st.session_state.q_item = int(np.random.randint(len(cat)))

        a, b = st.columns([3, 1])
        b.button("🎲 Random product", width="stretch", on_click=pick_random)
        i = a.selectbox("Product", range(len(cat)), format_func=lambda j: cat.at[j, "productDisplayName"],
                        key="q_item")
        q = cat.iloc[i]
        c1, c2 = st.columns([1, 5])
        c1.image(q["image_file"], width="stretch")
        c2.markdown(f"#### {q['productDisplayName']}\n{q['articleType']} · {q['baseColour']} · {q['gender']}")
        st.markdown("**Recommended** (✅ = same product type)")
        results(serve.item_query(PROPOSED, i), exclude=i, ref_type=q["articleType"])
