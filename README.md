# 🧵 CraftGuard
### AI-powered cultural design similarity detector for traditional Indian crafts

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Week%203%20Complete-green?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-655%20images-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## 🌍 The Problem

Traditional Indian crafts like **Kolhapuri chappals**, **Banarasi silk**, and **Ajrakh block prints** have been practiced by artisan communities for hundreds of years. Many hold **Geographical Indication (GI)** status — a legal protection similar to copyright.

Yet global fashion brands have repeatedly copied these designs without credit or compensation:
- **Prada** (2025) — sold $1,200 sandals nearly identical to Kolhapuri chappals
- **Gucci** (2019) — sold a Sikh turban as a fashion accessory for $790
- **Loewe** (2018) — used Ecuadorian indigenous textile patterns with no attribution

The artisans who created these traditions see none of the profit.

---

## 💡 The Solution

**CraftGuard** is a computer vision tool that:
1. Takes any fashion product image as input
2. Compares it against a database of traditional Indian craft patterns
3. Returns a **similarity score**, the **origin craft**, and a **GI status flag**

Built with CLIP embeddings + FAISS vector search — no expensive APIs, no subscriptions, fully open source.

---

## 🗂️ Project Roadmap

| Week | Task | Status |
|------|------|--------|
| 1 | Dataset collection — 655 images across 8 craft categories | ✅ Complete |
| 2 | CLIP embeddings + FAISS index | ✅ Complete |
| 3 | Streamlit web app + similarity search UI | ✅ Complete |
| 4 | Deploy to Hugging Face Spaces + write-up | ⏳ Upcoming |

---

## 📦 Week 1 — Dataset

### Craft categories collected

| Craft | State | GI Protected | Images |
|-------|-------|-------------|--------|
| Kolhapuri chappal | Maharashtra / Karnataka | ✅ Yes (2019) | 79 |
| Banarasi silk | Uttar Pradesh | ✅ Yes | 83 |
| Kalamkari | Andhra Pradesh | ✅ Yes | 78 |
| Ajrakh block print | Gujarat / Rajasthan | ✅ Yes | 74 |
| Ikat weave | Odisha / Telangana | ✅ Yes | 85 |
| Phulkari embroidery | Punjab | ✅ Yes | 99 |
| Chanderi fabric | Madhya Pradesh | ✅ Yes | 86 |
| Leheriya tie-dye | Rajasthan | ❌ No | 71 |

**Total: 655 images**

### Folder structure

```
craftguard/
├── data/
│   ├── images/
│   │   ├── kolhapuri/
│   │   ├── banarasi/
│   │   ├── kalamkari/
│   │   ├── ajrakh/
│   │   ├── ikat/
│   │   ├── phulkari/
│   │   ├── chanderi/
│   │   └── leheriya/
│   └── metadata.csv
├── notebooks/
│   ├── week1_dataset.ipynb
│   └── week2_embeddings.ipynb
└── README.md
```

### Metadata schema

Each image is logged in `metadata.csv` with the following fields:

```
image_id | craft_name | state | gi_status | source_url | license
```

### How we collected it

Used `icrawler` (Google Images → Bing fallback) to scrape images, then:
- Filtered out images smaller than 100×100px (logos, icons)
- Resized all images to **224×224px** (standard CLIP input size)
- Logged source URL and license per image

```python
# Install
pip install icrawler pillow tqdm pandas

# Run
python notebooks/week1_dataset.ipynb
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `icrawler` | Image scraping from Google/Bing |
| `Pillow` | Image resizing and cleaning |
| `pandas` | Metadata logging (CSV) |
| `CLIP` | Generate 512-dim image embeddings |
| `FAISS` | Fast cosine similarity search |
| `Streamlit` | Web app frontend |
| `Hugging Face Spaces` *(week 4)* | Free deployment |

---

## 🖥️ Week 3 — Streamlit Web App

### What it does

Upload any fashion product image → CraftGuard finds the 5 most similar traditional craft patterns from the database → returns similarity score, origin state, and GI protection status.

### How to run locally

```bash
git clone https://github.com/yourusername/craftguard
cd craftguard
python3 -m venv venv
source venv/bin/activate
pip install streamlit torch torchvision transformers faiss-cpu pillow pandas

# Mac users only — fixes OpenMP conflict between PyTorch and FAISS
KMP_DUPLICATE_LIB_OK=TRUE streamlit run app.py
```

### Results

The app correctly identifies craft similarity for close-up product images with scores typically in the 60–80% range for genuine matches.

| Query | Top Match | Score |
|-------|-----------|-------|
| Prada sandal close-up | Kolhapuri | 70.3% |
| Ajrakh print fabric | Ajrakh | ~85% |
| Random Nike shoe | Mixed low scores | <50% |

### Known limitations & future work

**Current limitation — whole image matching:**
CLIP analyses the entire uploaded image. If a fashion photo shows a full outfit (model on runway), it matches the overall visual style rather than isolating the specific craft element (e.g. the sandal).

**Fix planned for Week 4:**
Add a YOLO object detection step before CLIP to automatically crop the relevant region (footwear, textile, accessory) before running similarity search. This is the standard production approach for fine-grained retrieval.

```
Current:  full image → CLIP → similarity
Planned:  full image → YOLO crop → CLIP → similarity
```

**Other improvements planned:**
- Expand dataset beyond 8 crafts to 20+ categories
- Add more global craft databases (Ecuadorian textiles, African prints)
- Allow users to draw a bounding box on the uploaded image to select the region manually

---

## 🧠 Week 2 — CLIP Embeddings + FAISS Index

### How it works

Every image is converted into a list of 512 numbers called an **embedding** — a mathematical fingerprint of what the image looks like. Two visually similar images will have similar embeddings.

```
craft image → CLIP model → [0.23, 0.87, 0.12, ...] (512 numbers)
query image → CLIP model → [0.21, 0.85, 0.14, ...] (512 numbers)
                                    ↓
                         cosine similarity = 0.94 → very similar!
```

FAISS stores all 665 embeddings and searches through them in milliseconds.

### What we built

| File | Description |
|------|-------------|
| `data/embeddings.npy` | 665 × 512 matrix of CLIP embeddings |
| `data/embeddings_meta.csv` | Maps each embedding back to craft name + image path |
| `data/craftguard.index` | FAISS index for fast similarity search |

### Key numbers

- **665 images** embedded successfully
- **512 dimensions** per embedding (CLIP ViT-B/32)
- **< 1 second** to search all 665 embeddings via FAISS

### How to run

```python
pip install torch torchvision transformers faiss-cpu

# Generate embeddings
python notebooks/week2_embeddings.ipynb

# Query similarity
find_similar("your_image.jpg", top_k=5)
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/craftguard
cd craftguard
pip install -r requirements.txt
```

Open `notebooks/week1_dataset.ipynb` in Google Colab and run all cells.


---

## 📬 Contact

Built by Bhavini Kasana · (https://www.linkedin.com/in/bhavini-kasana-0b65151a9/) · [bhavini.kasana@edu.esiee.fr]

---

*This is a portfolio project built to demonstrate applied ML for social impact.*
