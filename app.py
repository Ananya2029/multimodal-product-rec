"""Multimodal product recommendation (image + text embeddings learned from scratch) - Streamlit app.

    python -m streamlit run app.py

Models are trained by colab/train_from_scratch.ipynb (src/scratch/); unzip its results into this folder.
The catalog shown here is the TEST split: products that no model saw during training.
"""
import json
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.config import RESULTS_DIR
from src.scratch import serve
from src.scratch.models import METHOD_NAMES

st.set_page_config(page_title="Multimodal Recommender", page_icon="🛍️", layout="wide")
SCRATCH_RES = serve.RES_DIR
METRIC_LABELS = {"i2i": "More like this (type)", "i2i_strict": "More like this (type + colour)",
                 "text2item": "Text query → product", "image2item": "Photo query → product"}


# ------------------------------------------------------------------ helpers
def read_csv(path):
    return pd.read_csv(path) if path.exists() else None


def mean_std(df, k):
    m, s = df.get(f"{k} mean"), df.get(f"{k} std")
    if m is None:
        return ["n/a"] * len(df)
    return ["n/a" if pd.isna(a) else f"{a:.3f}" + ("" if pd.isna(b) else f" ± {b:.3f}") for a, b in zip(m, s)]


def apply_filters(cat, s):
    mask = np.ones(len(cat), bool)
    if st.session_state.get("f_gender"):
        mask &= cat["gender"].isin(st.session_state.f_gender).values
    if st.session_state.get("f_cat"):
        mask &= cat["masterCategory"].isin(st.session_state.f_cat).values
    s = s.astype(float).copy()
    s[~mask] = -np.inf
    return s


def show_grid(cat, idx, scores, ref_type=None, cols=5):
    hits = [(i, s) for i, s in zip(idx, scores) if np.isfinite(s)]
    if not hits:
        st.info("No products match the current filters.")
        return
    columns = st.columns(cols)
    for n, (i, s) in enumerate(hits):
        r = cat.iloc[i]
        with columns[n % cols]:
            st.image(r["image_file"], width="stretch")
            mark = "" if ref_type is None else (" ✅" if r["articleType"] == ref_type else " ⚠️")
            st.markdown(f"**{r['productDisplayName']}**  \n<small>{r['articleType']} · {r['baseColour']} · "
                        f"{r['gender']}<br>score {s:.3f}{mark}</small>", unsafe_allow_html=True)


def recommend(method, q, k, exclude=None):
    if q is None:
        st.warning(f"{METHOD_NAMES[method]} can't use this kind of query; pick a multimodal model in the sidebar.")
        return None, None
    return serve.top_k(apply_filters(cat, serve.scores(method, q)), k, exclude)


def open_upload(upload):
    try:
        return Image.open(upload).convert("RGB")
    except Exception:
        st.error("Couldn't read that file as an image. Please upload a JPG, PNG or WEBP photo.")
        return None


# ------------------------------------------------------------------ sidebar
st.title("🛍️ Multimodal Product Recommendation")
st.caption("Image and text embeddings learned **from scratch**: a CNN for product photos and a Transformer for "
           "product titles, fused into one embedding. No pretrained weights.")
ready = serve.available()
if ready:
    cat = serve.catalog()
    methods = list(serve.export_info()["methods"])
    with st.sidebar:
        st.header("⚙️ Model")
        method = st.selectbox("Recommendation model", methods, index=methods.index("gated") if "gated" in methods else 0,
                              format_func=lambda m: METHOD_NAMES[m])
        k = st.slider("Number of results", 5, 30, 10, 5)
        st.header("🔎 Filters")
        st.multiselect("Gender", sorted(cat["gender"].dropna().unique()), key="f_gender")
        st.multiselect("Category", sorted(cat["masterCategory"].dropna().unique()), key="f_cat")
        info = serve.export_info()
        st.caption(f"Catalog: {len(cat):,} test products (never seen in training) · optimizer {info['optimizer']}")
    if "item" not in st.session_state:
        st.session_state.item = int(np.random.default_rng(0).integers(len(cat)))

PAGES = ["📊 Model & results", "🧥 More like this", "💬 Text search", "📷 Image search", "🧩 Image + text"]
page = st.radio("Section", PAGES, horizontal=True, label_visibility="collapsed", key="page")
st.divider()

if page != PAGES[0] and not ready:
    st.warning("The from-scratch models aren't trained yet. Run `colab/train_from_scratch.ipynb` on Google Colab, "
               "then unzip `mmrec_scratch_results.zip` into the project folder.")
    st.stop()

# ------------------------------------------------------------------ 1. model & results
if page == PAGES[0]:
    summary_f = SCRATCH_RES / "summary.json"
    if not summary_f.exists():
        st.warning("From-scratch results are not available yet: run `colab/train_from_scratch.ipynb` on Colab.")
    else:
        summ = json.loads(summary_f.read_text())
        if summ.get("epochs", 0) < 5:
            st.error("⚠️ These are **smoke-test** results (1 epoch on a tiny dataset), only to check the pipeline. "
                     "Run the Colab notebook for the real results.")
        meth = read_csv(SCRATCH_RES / "method_comparison.csv")
        st.subheader("🏆 Which fusion method gives the best recommendations?")
        st.markdown(f"All models are trained **from scratch** on {summ['n_products']:,} products and evaluated on "
                    f"**{summ['n_test']:,} unseen test products**. Scores are NDCG@10, as the mean ± std over "
                    f"{len(summ['seeds'])} random seeds (optimizer: **{summ['best_optimizer']}**, chosen on validation).")
        cards = st.columns(len(summ["winners"]))
        for col, (k_, w) in zip(cards, summ["winners"].items()):
            with col, st.container(border=True):
                st.caption(METRIC_LABELS[k_])
                st.metric(METHOD_NAMES[w["method"]], f"{w['ndcg']:.3f}")
        view = pd.DataFrame({"Method": meth["name"]})
        for k_, lab in METRIC_LABELS.items():
            view[lab] = mean_std(meth, k_)
        view["Params (M)"] = meth["params (M)"].round(2)
        st.dataframe(view, width="stretch", hide_index=True)
        if (SCRATCH_RES / "method_comparison.png").exists():
            st.image(str(SCRATCH_RES / "method_comparison.png"), width="stretch")

        sig = read_csv(SCRATCH_RES / "significance.csv")
        if sig is not None and len(sig):
            with st.expander("Statistical significance (paired bootstrap, proposed vs. each baseline)"):
                s = sig.copy()
                s["metric"] = s["metric"].map(METRIC_LABELS)
                s["proposed"] = s["proposed"].map(METHOD_NAMES)
                s["vs"] = s["vs"].map(METHOD_NAMES)
                s["significant (p<0.05)"] = np.where(s["p_value"] < 0.05, "✅", "–")
                st.dataframe(s[["metric", "proposed", "vs", "proposed_ndcg", "other_ndcg", "diff", "p_value",
                                "significant (p<0.05)"]].round(4), width="stretch", hide_index=True)

        opt = read_csv(SCRATCH_RES / "optimizer_comparison.csv")
        if opt is not None:
            st.subheader("⚙️ Optimizer comparison: SGD vs Adam vs AdamW")
            st.markdown("Proposed gated-fusion model. Each optimizer's learning rate is tuned on the validation "
                        "split, then trained with every seed.")
            ov = pd.DataFrame({"Optimizer": opt["optimizer"], "Learning rate": opt["lr"]})
            for k_, lab in METRIC_LABELS.items():
                ov[lab] = mean_std(opt, k_)
            ov["Train time (min)"] = opt["train min"].round(1)
            st.dataframe(ov, width="stretch", hide_index=True)
            for f in ("optimizer_curves.png", "optimizer_comparison.png"):
                if (SCRATCH_RES / f).exists():
                    st.image(str(SCRATCH_RES / f), width="stretch")

        abl = read_csv(SCRATCH_RES / "ablations.csv")
        if abl is not None:
            st.subheader("🧪 Ablation study")
            av = pd.DataFrame({"Variant": abl["name"]})
            for k_, lab in METRIC_LABELS.items():
                av[lab] = mean_std(abl, k_)
            st.dataframe(av, width="stretch", hide_index=True)

    # ---- GAN (trained from scratch)
    gan_json = RESULTS_DIR / "gan_metrics.json"
    if gan_json.exists():
        g = json.loads(gan_json.read_text())
        st.subheader("🎨 Conditional GAN: generating new product images")
        st.markdown("A conditional DCGAN trained from scratch generates 64×64 product images per category "
                    "(DiffAugment, minibatch-std, TTUR; Adam optimizer).")
        c1, c2 = st.columns(2)
        with c1, st.container(border=True):
            st.metric("FD-CLIP, generated vs real (lower is better)", f"{g['FD-CLIP (generated vs real)']:.3f}",
                      f"{g['FD-CLIP (generated vs real)'] - g['FD-CLIP untrained generator (baseline)']:+.3f} "
                      "vs untrained generator", delta_color="inverse")
            st.caption(f"Real vs real (best possible): {g['FD-CLIP real vs real (best possible)']:.3f}")
        with c2, st.container(border=True):
            st.metric("Generated images recognised as the requested category",
                      f"{g['class accuracy (generated)']:.0%}", f"chance = {g['chance accuracy']:.0%}",
                      delta_color="off")
            st.caption(f"Real held-out images: {g['class accuracy real held-out images']:.0%}")
        a, b = st.columns([1, 1])
        if (RESULTS_DIR / "gan_samples.png").exists():
            a.image(str(RESULTS_DIR / "gan_samples.png"), caption="Generated samples per category")
        if (RESULTS_DIR / "gan_losses.png").exists():
            b.image(str(RESULTS_DIR / "gan_losses.png"), caption="Training losses")

    # ---- reference only: pretrained models
    ref = RESULTS_DIR / "summary.json"
    if ref.exists():
        with st.expander("📎 Reference only: pretrained models (not used by our method)"):
            r = json.loads(ref.read_text())
            st.markdown("For comparison, the same kind of task was scored with **pretrained** encoders (CLIP, "
                        "SigLIP, DINOv2, MiniLM, ResNet-50, TF-IDF) on a 3,000-product sample. These models are "
                        "**not** part of our from-scratch method; they indicate an upper bound from "
                        f"large-scale pretraining. Best pretrained system: **{r['recommended']}**.")
            if (RESULTS_DIR / "model_comparison.png").exists():
                st.image(str(RESULTS_DIR / "model_comparison.png"), width="stretch")

    if ready:
        st.subheader("Side-by-side on one product")
        j = st.selectbox("Query product", range(len(cat)), index=st.session_state.item, key="cmp_item",
                         format_func=lambda x: cat.at[x, "productDisplayName"])
        picks = st.multiselect("Models", methods, default=[m for m in ("image", "text", "gated") if m in methods],
                               format_func=lambda m: METHOD_NAMES[m])
        st.image(cat.iloc[j]["image_file"], width=120)
        for m in picks:
            ii, ss = serve.top_k(serve.scores(m, serve.item_query(m, j)), 5, exclude=j)
            hits = (cat.iloc[ii]["articleType"] == cat.iloc[j]["articleType"]).mean()
            st.markdown(f"**{METHOD_NAMES[m]}** · {hits:.0%} same article type")
            show_grid(cat, ii, ss, ref_type=cat.iloc[j]["articleType"])

# ------------------------------------------------------------------ 2. more like this
elif page == PAGES[1]:
    c1, c2 = st.columns([3, 1])
    if c2.button("🎲 Random product", width="stretch"):
        st.session_state.item = int(np.random.randint(len(cat)))
    i = c1.selectbox("Pick a product", range(len(cat)), index=st.session_state.item,
                     format_func=lambda j: f"{cat.at[j, 'productDisplayName']}  (#{cat.at[j, 'id']})")
    q = cat.iloc[i]
    a, b = st.columns([1, 4])
    a.image(q["image_file"], width="stretch")
    b.subheader(q["productDisplayName"])
    b.write(f"{q['articleType']} · {q['baseColour']} · {q['gender']} · {q['usage']}")
    st.divider()
    idx, sc = recommend(method, serve.item_query(method, i), k, exclude=i)
    if idx is not None:
        st.markdown("**Recommended** (✅ = same article type as the query)")
        show_grid(cat, idx, sc, ref_type=q["articleType"])

# ------------------------------------------------------------------ 3. text search
elif page == PAGES[2]:
    text = st.text_input("Describe what you're looking for", "black handbag for women")
    if text.strip():
        idx, sc = recommend(method, serve.encode_query(method, text=text), k)
        if idx is not None:
            show_grid(cat, idx, sc)

# ------------------------------------------------------------------ 4. image search
elif page == PAGES[3]:
    up = st.file_uploader("Upload a product photo", type=["jpg", "jpeg", "png", "webp"], key="img_only")
    img = open_upload(up) if up else None
    if img is not None:
        st.image(img, width=180)
        idx, sc = recommend(method, serve.encode_query(method, image=img), k)
        if idx is not None:
            show_grid(cat, idx, sc)
    elif not up:
        st.info("Upload an image, e.g. a photo of a shoe, watch or t-shirt.")

# ------------------------------------------------------------------ 5. image + text
elif page == PAGES[4]:
    st.write("Combine a reference photo with words (uses the fused image + text embedding).")
    c1, c2 = st.columns([1, 2])
    up2 = c1.file_uploader("Reference image", type=["jpg", "jpeg", "png", "webp"], key="img_combo")
    mod = c2.text_input("Text", "red")
    img2 = open_upload(up2) if up2 else None
    if img2 is not None:
        c1.image(img2, width=160)
        idx, sc = recommend(method, serve.encode_query(method, image=img2, text=mod), k)
        if idx is not None:
            show_grid(cat, idx, sc)
