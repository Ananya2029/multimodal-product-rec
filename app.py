"""Multimodal product recommender - Streamlit app.

    streamlit run app.py
"""
import json
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


# keep at most 2 models in RAM (the DINOv2 + MiniLM system needs two); older ones are unloaded
@st.cache_resource(show_spinner="Loading model...", max_entries=2)
def encoder(name: str):
    if name == "TF-IDF":
        with open(EMB_DIR / "tfidf.pkl", "rb") as f:
            return pickle.load(f)
    return get_encoder(name)


@st.cache_resource(max_entries=32)  # cheap: embeddings are shared, see src.embeddings.load
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


def open_upload(upload):
    """Open an uploaded image; returns None (with a message) if it isn't a readable image."""
    try:
        return Image.open(upload).convert("RGB")
    except Exception:
        st.error("Couldn't read that file as an image. Please upload a JPG, PNG or WEBP photo.")
        return None


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

if "item" not in st.session_state:
    st.session_state.item = int(np.random.default_rng(0).integers(len(df)))
rec = recommender(sys_name, weight)
st.title("🛍️ Multimodal Product Recommendation")
st.caption(f"Using **{sys_name}**"
           + (f" · image weight {weight:.1f}" if system.group == "multimodal" else ""))

tab_cmp, tab_item, tab_text, tab_img, tab_combo = st.tabs(
    ["📊 Model comparison", "🧥 More like this", "💬 Text search", "📷 Image search", "🧩 Image + text"])

# ------------------------------------------------------------------ 1. model comparison (rendered first: it is the first tab and needs no model)
with tab_cmp:
    res_csv, summary_json = RESULTS_DIR / "model_comparison.csv", RESULTS_DIR / "summary.json"
    if not (res_csv.exists() and summary_json.exists()):
        st.warning("Run `python -m src.evaluate` to generate the comparison.")
    else:
        res = pd.read_csv(res_csv)
        summary = json.loads(summary_json.read_text())

        # ---- verdict
        st.subheader("🏆 Which model gives the best recommendations?")
        st.success(f"**Recommended: {summary['recommended']}**. It ranks best on average across all three "
                   f"tasks (evaluated on {len(df):,} products).")
        cards = st.columns(3)
        for col, t in zip(cards, summary["tasks"].values()):
            with col, st.container(border=True):
                st.caption(t["label"])
                st.metric(t["best"], f"{t['best_ndcg']:.3f}",
                          f"{t['best_ndcg'] - t['runner_up_ndcg']:+.3f} vs {t['runner_up']}")
                st.caption(f"95% CI {t['best_ci'][0]:.3f}–{t['best_ci'][1]:.3f} · "
                           f"p = {t['p_value']:.3f} · "
                           + ("✅ significant win" if t["significant"] else "≈ statistically tied")
                           + f" · {t['n_queries']:,} queries")

        # ---- overall ranking
        st.markdown("#### Overall ranking")
        overall = pd.DataFrame(summary["overall"])
        overall.insert(0, "#", range(1, len(overall) + 1))
        overall = overall.merge(res[["system", "group", "encode ms/item"]], on="system")
        overall.columns = ["#", "System", "Mean rank", "Rank: more like this", "Rank: type + colour",
                           "Rank: text search", "Type", "Encode ms/item (CPU)"]
        st.dataframe(overall, width="stretch", hide_index=True)

        with st.expander("How is this measured?"):
            st.markdown(
                "- **More like this**: each of the catalog products is used as a query; a recommendation is "
                "relevant if it has the same article type (strict: same type **and** colour).\n"
                "- **Text search**: shopper queries such as *“navy blue shirts for men”*; relevant = same "
                "colour + type + gender.\n"
                "- **NDCG@10** rewards putting relevant products near the top of the 10 results (1.0 = perfect).\n"
                "- **95% CI** comes from bootstrap resampling of the queries. The **p-value** comes from a paired "
                "bootstrap test of the winner against the runner-up; p < 0.05 means the win is not due to chance.\n"
                "- Systems that cannot handle a task (ResNet/DINOv2 cannot read text) rank last on it.")

        # ---- details
        st.markdown("#### Detailed metrics")
        show = ["system", "group"] + [f"{t} {m}" for t in ("i2i", "i2i-strict", "search")
                                      for m in ("P@10", "mAP@10", "NDCG@10")]
        show = [c for c in show if c in res.columns]
        st.dataframe(res[show].style
                     .background_gradient(subset=show[2:], cmap="Greens")
                     .highlight_max(subset=show[2:], props="font-weight: bold")
                     .format(precision=3, na_rep="n/a"), width="stretch", hide_index=True)
        st.image(str(RESULTS_DIR / "model_comparison.png"), width="stretch",
                 caption="NDCG@10 with 95% confidence intervals (★ = best)")
        st.markdown("#### Image vs text weight")
        st.image(str(RESULTS_DIR / "fusion_sweep.png"), width="stretch",
                 caption="Mixing modalities beats either one alone; the best image weight is about 0.3–0.6")

    st.subheader("Side-by-side on one product")
    st.caption("A single product is only an illustration; the metrics above average over all 3,000. Titles that "
               "share a brand (e.g. *Jealous 21*) pull text-based models toward that brand's other items, a known "
               "limitation. Try a few products to see where each model succeeds or fails.")
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

# ------------------------------------------------------------------ 2. item-to-item
with tab_item:
    c1, c2 = st.columns([3, 1])
    options = df.index.tolist()
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

# ------------------------------------------------------------------ 3. text search
with tab_text:
    text = st.text_input("Describe what you're looking for", "black leather handbag for women")
    st.caption("Tip: CLIP/SigLIP image models search the *pictures* directly (zero-shot); "
               "text models search product titles.")
    if text:
        idx, sc = recommend(rec, df, k, **encode_query(system, None, text))
        if idx is not None:
            show_grid(df, idx, sc)

# ------------------------------------------------------------------ 4. image search
with tab_img:
    up = st.file_uploader("Upload a product photo", type=["jpg", "jpeg", "png", "webp"], key="img_only")
    img = open_upload(up) if up else None
    if img is not None:
        st.image(img, width=180)
        idx, sc = recommend(rec, df, k, **encode_query(system, img, None))
        if idx is not None:
            show_grid(df, idx, sc)
    elif not up:
        st.info("Upload an image, e.g. a photo of a shoe, watch or t-shirt.")

# ------------------------------------------------------------------ 5. composed query
with tab_combo:
    st.write("Combine a reference image with a text modifier (works best with CLIP / SigLIP).")
    c1, c2 = st.columns([1, 2])
    up2 = c1.file_uploader("Reference image", type=["jpg", "jpeg", "png", "webp"], key="img_combo")
    mod = c2.text_input("Text modifier", "in red colour")
    img2 = open_upload(up2) if up2 else None
    if img2 is not None:
        c1.image(img2, width=160)
        idx, sc = recommend(rec, df, k, **encode_query(system, img2, mod))
        if idx is not None:
            show_grid(df, idx, sc)
