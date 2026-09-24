"""Multimodal product recommender - Streamlit app.

    python -m streamlit run app.py
"""
import json
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # hide transformers deprecation spam

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src import query as Q
from src.config import DEFAULT_IMAGE_WEIGHT, IMAGE_DIR, RESULTS_DIR, ROOT
from src.data import load_catalog
from src.recommender import SYSTEMS_BY_NAME, Recommender, available_systems

MODEL_DIR = ROOT / "models"
st.set_page_config(page_title="Multimodal Recommender", page_icon="🛍️", layout="wide")


# ------------------------------------------------------------------ cached resources
@st.cache_data
def catalog() -> pd.DataFrame:
    return load_catalog()


# keep at most 2 models in RAM (DINOv2 + MiniLM and the trained heads need two); older ones are unloaded
@st.cache_resource(show_spinner="Loading model...", max_entries=2)
def encoder(name: str):
    return Q.load_model(name)


@st.cache_resource(max_entries=32)  # cheap: embeddings are shared, see src.embeddings.load
def recommender(system_name: str, weight: float):
    return Recommender(SYSTEMS_BY_NAME[system_name], weight)


def encode_query(system, image: Image.Image | None, text: str | None) -> dict:
    """Embed a free-form query for every side of the catalog the system can score (see src/query.py)."""
    try:
        return Q.encode_query(system, image, text, encoder)
    except MemoryError as e:  # low-RAM guard in src/encoders.py: warn instead of crashing the app
        st.warning(f"⚠️ {e}")
        st.stop()


@st.cache_data(show_spinner="Downloading image ...", max_entries=20, ttl=3600)
def fetch_url_image(url: str) -> Image.Image:
    from src.web import fetch_image
    return fetch_image(url)


@st.cache_resource
def live_catalog():
    from src.live import load_live_catalog
    return load_live_catalog()


@st.cache_resource
def gan_generator():
    from src.gan import load_generator
    return load_generator()


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
names = [s.name for s in available_systems()]
with st.sidebar:
    st.header("⚙️ Model")
    sys_name = st.selectbox("Recommendation system", names, index=names.index("CLIP image + text"),
                            help="Text-only, image-only, multimodal (fused) or trained fusion embeddings")
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
           + (f" · image weight {weight:.1f}" if system.group == "multimodal" else "")
           + (" · trained on 70% of the catalog (see the optimizer comparison)" if system.group == "trained" else ""))

# Navigation: unlike st.tabs (which runs every tab on every click), only the selected section runs,
# so models are loaded only when a section needs them (important on machines with little RAM).
PAGES = ["📊 Model comparison", "🧥 More like this", "💬 Text search", "📷 Image search", "🧩 Image + text",
         "🌐 Internet", "🎨 GAN"]
page = st.radio("Section", PAGES, horizontal=True, label_visibility="collapsed", key="page")
st.divider()

# ------------------------------------------------------------------ 1. model comparison (rendered first: it is the first tab and needs no model)
if page == PAGES[0]:
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

    # ---- trained fusion head: optimizer comparison
    opt_csv = RESULTS_DIR / "optimizer_comparison.csv"
    if opt_csv.exists():
        st.subheader("🧠 Training a fusion model: SGD vs Adam vs AdamW")
        st.markdown(
            "A small neural network is trained on top of frozen CLIP image + text embeddings (supervised "
            "contrastive loss). Each optimizer is **tuned on a validation split**, then trained with **3 seeds**. "
            "Scores are on **900 test products never seen in training** (mean ± std).")
        opt = pd.read_csv(opt_csv)
        view = pd.DataFrame({"Model": opt["model"],
                             "Hyper-parameters": opt["hyper-parameters"].fillna("-")})
        for c, label in (("i2i NDCG@10", "More like this"), ("i2i-strict NDCG@10", "Type + colour"),
                         ("search NDCG@10", "Text search")):
            view[label] = [("n/a" if pd.isna(m) else f"{m:.3f}" + ("" if pd.isna(sd) else f" ± {sd:.3f}"))
                           for m, sd in zip(opt[c], opt[f"{c} std"])]
        view["Train time (s)"] = opt["train time (s)"].round(0)
        st.dataframe(view, width="stretch", hide_index=True)
        trained = opt[opt["optimizer"] != "-"]
        base = opt.iloc[0]
        best = trained.loc[trained["search NDCG@10"].idxmax()]
        st.info(f"**Training helps much more than the optimizer choice:** every optimizer lifts 'more like this' "
                f"from {base['i2i NDCG@10']:.3f} (zero-shot CLIP) to about {trained['i2i NDCG@10'].mean():.3f}. "
                f"Between optimizers the differences are small; **{best['optimizer']}** is best on text search "
                f"({best['search NDCG@10']:.3f}). Adam and AdamW behave almost identically here.")
        if (RESULTS_DIR / "optimizer_curves.png").exists():
            st.image(str(RESULTS_DIR / "optimizer_curves.png"), width="stretch")

    # ---- GAN
    gan_json = RESULTS_DIR / "gan_metrics.json"
    if gan_json.exists():
        g = json.loads(gan_json.read_text())
        st.subheader("🎨 Conditional GAN (DCGAN): how good are the generated products?")
        c1, c2 = st.columns(2)
        with c1, st.container(border=True):
            st.metric("FD-CLIP, generated vs real (lower is better)", f"{g['FD-CLIP (generated vs real)']:.3f}",
                      f"{g['FD-CLIP (generated vs real)'] - g['FD-CLIP untrained generator (baseline)']:+.3f} "
                      "vs untrained generator", delta_color="inverse")
            st.caption(f"Best possible (real vs real): {g['FD-CLIP real vs real (best possible)']:.3f}")
        with c2, st.container(border=True):
            st.metric("Generated images recognised as the requested category",
                      f"{g['class accuracy (generated)']:.0%}",
                      f"chance = {g['chance accuracy']:.0%}", delta_color="off")
            st.caption(f"Real held-out images: {g['class accuracy real held-out images']:.0%} · "
                       f"untrained generator: {g['class accuracy untrained generator']:.0%}")
        st.caption(f"{g['n_images']:,} training images · {g['n_classes']} categories · {g['epochs']} epochs · "
                   f"{g['train_minutes']} min on CPU · optimizer: Adam (lr 2e-4, betas 0.5/0.999)")

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
if page == PAGES[1]:
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
if page == PAGES[2]:
    text = st.text_input("Describe what you're looking for", "black leather handbag for women")
    st.caption("Tip: CLIP/SigLIP image models search the *pictures* directly (zero-shot); "
               "text models search product titles.")
    if text:
        idx, sc = recommend(rec, df, k, **encode_query(system, None, text))
        if idx is not None:
            show_grid(df, idx, sc)

# ------------------------------------------------------------------ 4. image search
if page == PAGES[3]:
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
if page == PAGES[4]:
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

# ------------------------------------------------------------------ 6. internet: URL images + live shop API
if page == PAGES[5]:
    st.subheader("🔗 Search with an image from the internet")
    st.caption("Paste a direct link to a product photo (.jpg / .png / .webp) from any website.")
    c1, c2 = st.columns([3, 1])
    url = c1.text_input("Image URL", placeholder="https://.../photo.jpg", key="img_url")
    url_text = c2.text_input("Optional text", placeholder="e.g. in black", key="img_url_text")
    if url.strip():
        try:
            web_img = fetch_url_image(url.strip())
        except Exception as e:
            web_img = None
            st.error(str(e))
        if web_img is not None:
            st.image(web_img, width=160)
            if Q.can_handle(system, True, bool(url_text.strip())):
                idx, sc = recommend(rec, df, k, **encode_query(system, web_img, url_text))
                if idx is not None:
                    show_grid(df, idx, sc)
            else:
                st.info(f"{sys_name} can't take an image query; pick an image-capable model in the sidebar.")

    st.divider()
    st.subheader("🛒 Live products from an online shop API")
    st.caption("Products are fetched from the public API **dummyjson.com/products** and embedded with CLIP, so "
               "they can be searched and matched against our fashion catalog.")
    cat = live_catalog()
    if st.button("🔄 Fetch products from the internet" if cat is None else "🔄 Refresh from the internet"):
        from src.live import build_live_catalog
        bar = st.progress(0.0, "Contacting dummyjson.com ...")
        try:
            build_live_catalog(clip=encoder("CLIP"), progress=lambda f, m: bar.progress(min(f, 1.0), m))
            live_catalog.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Couldn't fetch the live catalog: {e}. Check your internet connection.")
    if cat is not None:
        from src.live import image_path, live_scores
        prods = cat["products"]
        st.caption(f"{len(prods)} products · {len({p['category'] for p in prods})} categories · "
                   f"fetched {cat['fetched_at']}")
        live_q = st.text_input("Search the live shop", placeholder="e.g. black sunglasses", key="live_q")
        if live_q.strip():
            try:
                s_live = live_scores(cat, q_txt=encoder("CLIP").encode_text([live_q])[0])
            except MemoryError as e:
                st.warning(f"⚠️ {e}")
                st.stop()
            top = np.argsort(-s_live)[:k]
            cols = st.columns(5)
            for n, i in enumerate(top):
                p = prods[i]
                with cols[n % 5]:
                    st.image(str(image_path(p)), width="stretch")
                    st.markdown(f"**{p['title']}**  \n<small>{p['category']} · ${p['price']} · "
                                f"score {s_live[i]:.3f}</small>", unsafe_allow_html=True)
        st.markdown("##### Match a live product against our fashion catalog")
        default_pick = next((i for i, p in enumerate(prods) if "shirt" in p["category"]), 0)
        pick = st.selectbox("Live product", range(len(prods)), key="live_pick", index=default_pick,
                            format_func=lambda i: f"{prods[i]['title']} ({prods[i]['category']})")
        a, b = st.columns([1, 4])
        a.image(str(image_path(prods[pick])), width="stretch")
        clip_rec = recommender("CLIP image + text", DEFAULT_IMAGE_WEIGHT)
        idx, sc = recommend(clip_rec, df, k, q_img=cat["I"][pick], q_txt=cat["T"][pick])
        with b:
            st.caption("Most similar products in our catalog (CLIP image + text):")
            if idx is not None:
                show_grid(df, idx, sc)

# ------------------------------------------------------------------ 7. GAN
if page == PAGES[6]:
    st.subheader("🎨 Design a new product with a GAN")
    if not (MODEL_DIR / "gan_generator.pt").exists():
        st.warning("The GAN hasn't been trained yet: run `python -m src.gan`.")
    else:
        from src.gan import generate
        G, classes = gan_generator()
        st.caption("A conditional DCGAN trained on our product photos generates new 64×64 concept images for a "
                   "category; the recommender then finds the most similar real products.")
        c1, c2, c3 = st.columns([2, 1, 1])
        gan_cls = c1.selectbox("Category", classes, index=classes.index("Watches") if "Watches" in classes else 0)
        n_gen = c2.slider("Images", 1, 8, 6)
        gan_seed = c3.number_input("Random seed", 0, 10_000, 0, help="Change it to get different designs")
        gen = generate(G, classes.index(gan_cls), n_gen, seed=int(gan_seed))
        cols = st.columns(n_gen)
        for n, im in enumerate(gen):
            cols[n].image(im.resize((128, 128), Image.NEAREST), caption=f"#{n + 1}", width="stretch")
        which = st.radio("Find real products similar to design", list(range(1, n_gen + 1)), horizontal=True,
                         format_func=lambda v: f"#{v}") - 1
        if Q.can_handle(system, True, False):
            idx, sc = recommend(rec, df, k, **encode_query(system, gen[which], None))
            if idx is not None:
                st.caption(f"Similar real products (using {sys_name}):")
                show_grid(df, idx, sc)
        else:
            st.info(f"{sys_name} can't take an image query; pick an image-capable model in the sidebar.")
        if (RESULTS_DIR / "gan_losses.png").exists():
            with st.expander("Training details"):
                st.image(str(RESULTS_DIR / "gan_losses.png"), width="stretch")
                if (RESULTS_DIR / "gan_samples.png").exists():
                    st.image(str(RESULTS_DIR / "gan_samples.png"), caption="Samples for every category")
