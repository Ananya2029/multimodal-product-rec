"""Multimodal product recommender - Streamlit app.

    streamlit run app.py
"""
import os
import pickle

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # hide transformers deprecation spam

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.config import DEFAULT_IMAGE_WEIGHT, EMB_DIR, IMAGE_DIR, RESULTS_DIR
from src.data import load_catalog
from src.encoders import get_encoder
from src.recommender import SYSTEMS, SYSTEMS_BY_NAME, Recommender

SHARED = ("CLIP", "SigLIP")
st.set_page_config(page_title="Multimodal Recommender", page_icon="🛍️", layout="wide")


# ------------------------------------------------------------------ cached resources
@st.cache_data
def catalog() -> pd.DataFrame:
    return load_catalog()


@st.cache_resource(show_spinner="Loading model...")
def encoder(name: str):
    if name == "TF-IDF":
        with open(EMB_DIR / "tfidf.pkl", "rb") as f:
            return pickle.load(f)
    return get_encoder(name)


@st.cache_resource
def recommender(system_name: str, weight: float):
    return Recommender(SYSTEMS_BY_NAME[system_name], weight)


def encode_query(system, image: Image.Image | None, text: str | None) -> dict:
    """Embed a free-form query for every side of the catalog the system can score."""
    q = {}
    text = (text or "").strip() or None
    if image is not None:
        if system.image_model:
            q["q_img"] = encoder(system.image_model).encode_image([image])[0]
        if system.text_model in SHARED:
            q["q_img_textenc"] = encoder(system.text_model).encode_image([image])[0]
    if text:
        if system.text_model:
            q["q_txt"] = encoder(system.text_model).encode_text([text])[0]
        if system.image_model in SHARED:
            q["q_txt_imgenc"] = encoder(system.image_model).encode_text([text])[0]
    return q


# ------------------------------------------------------------------ UI helpers
def apply_filters(df, scores):
    mask = np.ones(len(df), bool)
    if st.session_state.get("f_gender"):
        mask &= df["gender"].isin(st.session_state.f_gender).values
    if st.session_state.get("f_cat"):
        mask &= df["masterCategory"].isin(st.session_state.f_cat).values
    s = scores.astype(float).copy()
    s[~mask] = -np.inf
    return s


def show_grid(df, idx, scores, ref_type=None, cols=5):
    hits = [(i, s) for i, s in zip(idx, scores) if np.isfinite(s)]
    if not hits:
        st.info("No products match the current filters.")
        return
    columns = st.columns(cols)
    for n, (i, s) in enumerate(hits):
        row = df.iloc[i]
        with columns[n % cols]:
            st.image(str(IMAGE_DIR / row["image_path"]), width="stretch")
            match = "" if ref_type is None else (" ✅" if row["articleType"] == ref_type else " ⚠️")
            st.markdown(f"**{row['productDisplayName']}**  \n"
                        f"<small>{row['articleType']} · {row['baseColour']} · {row['gender']}"
                        f"<br>score {s:.3f}{match}</small>", unsafe_allow_html=True)


def recommend(rec, df, k, exclude=None, **query):
    scores = rec.query_scores(**query)
    if scores is None:
        st.warning("This model can't handle that kind of query. Pick a text-capable or image-capable model.")
        return None, None
    return rec.top_k(apply_filters(df, scores), k, exclude=exclude)


# ------------------------------------------------------------------ sidebar
df = catalog()
names = [s.name for s in SYSTEMS]
with st.sidebar:
    st.header("⚙️ Model")
    sys_name = st.selectbox("Recommendation system", names, index=names.index("CLIP image + text"),
                            help="Text-only, image-only or multimodal (fused) embeddings")
    system = SYSTEMS_BY_NAME[sys_name]
    weight = 1.0 if system.text_model is None else 0.0 if system.image_model is None else \
        st.slider("Image ↔ text weight", 0.0, 1.0, DEFAULT_IMAGE_WEIGHT, 0.1,
                  help="1.0 = image embedding only, 0.0 = text embedding only")
    k = st.slider("Number of results", 5, 30, 10, 5)
    st.header("🔎 Filters")
    st.multiselect("Gender", sorted(df["gender"].dropna().unique()), key="f_gender")
    st.multiselect("Category", sorted(df["masterCategory"].dropna().unique()), key="f_cat")
    st.caption(f"Catalog: {len(df):,} products · {df['articleType'].nunique()} article types")

rec = recommender(sys_name, weight)
st.title("🛍️ Multimodal Product Recommendation")
st.caption(f"Using **{sys_name}**"
           + (f" · image weight {weight:.1f}" if system.group == "multimodal" else ""))

tab_item, tab_text, tab_img, tab_combo, tab_cmp = st.tabs(
    ["🧥 More like this", "💬 Text search", "📷 Image search", "🧩 Image + text", "📊 Model comparison"])

# ------------------------------------------------------------------ 1. item-to-item
with tab_item:
    c1, c2 = st.columns([3, 1])
    options = df.index.tolist()
    if "item" not in st.session_state:
        st.session_state.item = int(np.random.default_rng(0).integers(len(df)))
    if c2.button("🎲 Random product", width="stretch"):
        st.session_state.item = int(np.random.randint(len(df)))
    i = c1.selectbox("Pick a product", options, index=st.session_state.item,
                     format_func=lambda j: f"{df.at[j, 'productDisplayName']}  (#{df.at[j, 'id']})")
    q = df.iloc[i]
    a, b = st.columns([1, 4])
    a.image(str(IMAGE_DIR / q["image_path"]), width="stretch")
    b.subheader(q["productDisplayName"])
    b.write(f"{q['articleType']} · {q['baseColour']} · {q['gender']} · {q['usage']}")
    st.divider()
    # the catalog item's own cached embeddings serve as the query
    query = {}
    if rec.I is not None:
        query["q_img"] = rec.I[i]
    if rec.T is not None:
        query["q_txt"] = rec.T[i]
    idx, sc = recommend(rec, df, k, exclude=i, **query)
    if idx is not None:
        st.markdown(f"**Recommended** (✅ = same article type as the query)")
        show_grid(df, idx, sc, ref_type=q["articleType"])

# ------------------------------------------------------------------ 2. text search
with tab_text:
    text = st.text_input("Describe what you're looking for", "black leather handbag for women")
    st.caption("Tip: CLIP/SigLIP image models search the *pictures* directly (zero-shot); "
               "text models search product titles.")
    if text:
        idx, sc = recommend(rec, df, k, **encode_query(system, None, text))
        if idx is not None:
            show_grid(df, idx, sc)

# ------------------------------------------------------------------ 3. image search
with tab_img:
    up = st.file_uploader("Upload a product photo", type=["jpg", "jpeg", "png", "webp"], key="img_only")
    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, width=180)
        idx, sc = recommend(rec, df, k, **encode_query(system, img, None))
        if idx is not None:
            show_grid(df, idx, sc)
    else:
        st.info("Upload an image, e.g. a photo of a shoe, watch or t-shirt.")

# ------------------------------------------------------------------ 4. composed query
with tab_combo:
    st.write("Combine a reference image with a text modifier (works best with CLIP / SigLIP).")
    c1, c2 = st.columns([1, 2])
    up2 = c1.file_uploader("Reference image", type=["jpg", "jpeg", "png", "webp"], key="img_combo")
    mod = c2.text_input("Text modifier", "in red colour")
    if up2:
        img = Image.open(up2).convert("RGB")
        c1.image(img, width=160)
        idx, sc = recommend(rec, df, k, **encode_query(system, img, mod))
        if idx is not None:
            show_grid(df, idx, sc)

# ------------------------------------------------------------------ 5. comparison
with tab_cmp:
    res_csv = RESULTS_DIR / "model_comparison.csv"
    if not res_csv.exists():
        st.warning("Run `python -m src.evaluate` to generate the comparison.")
    else:
        res = pd.read_csv(res_csv)
        st.subheader("Offline evaluation")
        st.markdown("- **i2i**: *more like this*; relevant = same article type\n"
                    "- **i2i-strict**: same article type **and** colour\n"
                    "- **search**: shopper queries like *“navy blue shirts for men”*; "
                    "relevant = same colour + type + gender")
        metric_cols = [c for c in res.columns if "NDCG" in c or "mAP" in c or "ms/item" in c]
        st.dataframe(res[["system", "group"] + metric_cols].style
                     .background_gradient(subset=[c for c in metric_cols if "ms" not in c], cmap="Greens")
                     .format(precision=3), width="stretch", hide_index=True)
        for png in ("model_comparison.png", "fusion_sweep.png"):
            if (RESULTS_DIR / png).exists():
                st.image(str(RESULTS_DIR / png), width="stretch")

    st.subheader("Side-by-side on one product")
    j = st.selectbox("Query product", df.index.tolist(), index=st.session_state.item, key="cmp_item",
                     format_func=lambda x: df.at[x, "productDisplayName"])
    picks = st.multiselect("Systems", names, default=["TF-IDF (title)", "DINOv2 image", "CLIP image + text"])
    st.image(str(IMAGE_DIR / df.iloc[j]["image_path"]), width=120)
    for name in picks:
        r = recommender(name, DEFAULT_IMAGE_WEIGHT)
        qq = {**({"q_img": r.I[j]} if r.I is not None else {}), **({"q_txt": r.T[j]} if r.T is not None else {})}
        ii, ss = r.top_k(r.query_scores(**qq), 5, exclude=j)
        hits = (df.iloc[ii]["articleType"] == df.iloc[j]["articleType"]).mean()
        st.markdown(f"**{name}** · {hits:.0%} same article type")
        show_grid(df, ii, ss, ref_type=df.iloc[j]["articleType"], cols=5)
