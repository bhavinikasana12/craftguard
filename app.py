import streamlit as st
import torch
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

st.set_page_config(page_title="CraftGuard", page_icon="🧵", layout="centered")

BASE_DIR = os.path.expanduser("~/craftguard/data")

GI_INFO = {
    "kolhapuri":  {"gi": "✅ GI Protected (2019)", "state": "Maharashtra / Karnataka"},
    "banarasi":   {"gi": "✅ GI Protected",         "state": "Uttar Pradesh"},
    "kalamkari":  {"gi": "✅ GI Protected",         "state": "Andhra Pradesh"},
    "ajrakh":     {"gi": "✅ GI Protected",         "state": "Gujarat / Rajasthan"},
    "ikat":       {"gi": "✅ GI Protected",         "state": "Odisha / Telangana"},
    "phulkari":   {"gi": "✅ GI Protected",         "state": "Punjab"},
    "chanderi":   {"gi": "✅ GI Protected",         "state": "Madhya Pradesh"},
    "leheriya":   {"gi": "❌ Not GI Protected",     "state": "Rajasthan"},
}

@st.cache_resource
def load_everything():
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    index   = faiss.read_index(f"{BASE_DIR}/craftguard.index")
    meta_df = pd.read_csv(f"{BASE_DIR}/embeddings_meta.csv")
    return model, processor, index, meta_df

model, processor, index, meta_df = load_everything()

def embed_image(img):
    inputs = processor(images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        raw = model.get_image_features(**inputs)
        emb = raw.pooler_output if hasattr(raw, "pooler_output") else raw
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().numpy().astype("float32").reshape(1, -1)

st.title("🧵 CraftGuard")
st.markdown("### AI-powered cultural design similarity detector")
st.markdown("Upload any fashion product image to check if it resembles a traditional Indian craft pattern.")
st.divider()

uploaded = st.file_uploader("Upload a fashion product image", type=["jpg", "jpeg", "png"])

if uploaded:
    query_img = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(query_img, caption="Your uploaded image", width="stretch")
    with col2:
        with st.spinner("Analysing design..."):
            query_emb = embed_image(query_img)
            scores, indices = index.search(query_emb, 6)
        st.markdown("#### Top matches")
        for rank, (score, idx) in enumerate(zip(scores[0][1:], indices[0][1:]), 1):
            row     = meta_df.iloc[idx]
            craft   = row["craft_name"]
            info    = GI_INFO.get(craft, {"gi": "Unknown", "state": "Unknown"})
            sim_pct = round(float(score) * 100, 1)
            if sim_pct < 78:
                continue  # skip low confidence matches
            with st.expander(f"#{rank} — {craft.title()}  ({sim_pct}% similar)", expanded=rank==1):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.image(Image.open(row["path"]), width="stretch")
                with c2:
                    st.markdown(f"**Craft:** {craft.title()}")
                    st.markdown(f"**Origin:** {info['state']}")
                    st.markdown(f"**Status:** {info['gi']}")
                    st.progress(sim_pct / 100)
                    st.caption(f"Similarity score: {sim_pct}%")

    st.divider()
    top_craft = meta_df.iloc[indices[0][1]]["craft_name"]
    top_score = float(scores[0][1]) * 100
    top_gi    = GI_INFO.get(top_craft, {}).get("gi", "")
    if top_score > 82 and "✅" in top_gi:
        st.error(f"⚠️ High similarity detected! This design is {top_score:.1f}% similar to {top_craft.title()} — a GI-protected craft. Commercial use without attribution may violate GI protections.")
    elif top_score > 78 and "✅" in top_gi:
        st.warning(f"🔍 Moderate similarity to {top_craft.title()} ({top_score:.1f}%). Review recommended.")
    else:
        st.success("✅ No significant craft similarity detected. Scores below 78% are not considered a match.")
else:
    st.info("👆 Upload an image above to get started.")
    cols = st.columns(4)
    for i, craft in enumerate(GI_INFO.keys()):
        cols[i % 4].markdown(f"• {craft.title()}")
