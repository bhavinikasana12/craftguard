#  CraftGuard

### AI-powered cultural design similarity detector for traditional Indian crafts


---

##  The Problem

Traditional Indian crafts like **Kolhapuri chappals**, **Banarasi silk**, and **Ajrakh block prints** have been practiced by artisan communities for hundreds of years. Many hold **Geographical Indication (GI)** status — a legal protection recognising the cultural and economic rights of the communities who created them.

Yet global fashion brands have repeatedly copied these designs without credit or compensation:

- **Prada** (2025) — sold $1,200 sandals nearly identical to Kolhapuri chappals
- **Gucci** (2019) — sold a Sikh turban as a fashion accessory for $790
- **Loewe** (2018) — used Ecuadorian indigenous textile patterns with no attribution

The artisans who created these traditions see none of the profit.

---

##  The Research Gap

Existing tools in this space focus on **archiving and digitising** traditional designs:

| Tool | Organisation | What it does |
|------|-------------|--------------|
| VisioNxt | Ministry of Textiles + NIFT | AI trend forecasting for Indian fashion |
| DigiBunai | IIT Guwahati | Digitises traditional weaving patterns |
| TCS Intelligent Saree Platform | Tata Consultancy Services | Digitises Kanjivaram saree designs |
| Kosha.ai | Private | IoT-based product authentication |

**None of these tools detect when a commercial fashion product visually copies a traditional craft pattern.** There is currently no open-source model trained specifically to distinguish between Indian craft categories such as Ajrakh, Banarasi, Phulkari, or Ikat.

CraftGuard is an attempt to address this gap.

---

##  The Solution

**CraftGuard** is a computer vision tool that:

1. Takes any fashion product image as input
2. Compares it against a database of 655 traditional Indian craft images
3. Returns a **similarity score**, the **origin craft**, and a **GI status flag**

Built with CLIP embeddings + FAISS vector search.

---

##  How It Works — CLIP + FAISS Explained

### What is CLIP?

CLIP (Contrastive Language-Image Pretraining) is a model developed by OpenAI, trained on 400 million image-text pairs from the internet. It converts any image into a list of 512 numbers — called an **embedding** — that captures the visual meaning of the image.

```
Kolhapuri sandal image → CLIP → [0.23, 0.87, 0.12, ...]  512 numbers
Prada sandal image     → CLIP → [0.21, 0.85, 0.14, ...]  512 numbers
                                          ↑
                               similar numbers = similar images
```

Two visually similar images produce similar embeddings. This allows CraftGuard to compare an unknown fashion product against every craft image in the database — without needing to label or describe the query image.

### What is FAISS?

FAISS (Facebook AI Similarity Search) is a library developed by Meta for searching through large collections of vectors efficiently. Without FAISS, finding the most similar image in a database of 655 embeddings would require 655 individual comparisons. At millions of images, this becomes impractically slow.

FAISS organises all embeddings into an index and searches through them in milliseconds — making it the natural pairing for CLIP in similarity search applications.

```
655 craft images
    → CLIP converts each to 512 numbers
        → FAISS stores all 655 × 512 numbers in an index
            → User uploads image
                → CLIP converts it to 512 numbers
                    → FAISS searches index in < 1 second
                        → Returns top 5 most similar craft images
```

---

##  Project Roadmap

| Week | Task | Status |
|------|------|--------|
| 1 | Dataset collection — 655 images across 8 craft categories | ✅ Complete |
| 2 | CLIP embeddings + FAISS index | ✅ Complete |
| 3 | Streamlit web app + similarity search UI | ✅ Complete |


---

##  Dataset

### Craft categories

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

### Collection method

Used `icrawler` (Google Images → Bing fallback) to scrape images, then:

- Filtered out images smaller than 100×100px
- Resized all images to **224×224px** (standard CLIP input size)
- Logged source URL and license per image in `metadata.csv`

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
│   ├── embeddings.npy
│   ├── embeddings_meta.csv
│   ├── craftguard.index
│   └── metadata.csv
├── notebooks/
│   ├── week1_dataset.ipynb
│   └── week2_embeddings.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

##  Tech Stack

| Tool | Purpose |
|------|---------|
| `icrawler` | Image scraping from Google/Bing |
| `Pillow` | Image resizing and cleaning |
| `pandas` | Metadata logging (CSV) |
| `CLIP` (ViT-B/32) | Generate 512-dim image embeddings |
| `FAISS` | Fast cosine similarity search |
| `Streamlit` | Web app frontend |


---

##  Running Locally

```bash
git clone https://github.com/bhavinikasana12/craftguard
cd craftguard
python3 -m venv venv
source venv/bin/activate
pip install streamlit torch torchvision transformers faiss-cpu pillow pandas

# Mac users only — fixes OpenMP conflict between PyTorch and FAISS
KMP_DUPLICATE_LIB_OK=TRUE streamlit run app.py
```

---

##  Results

CraftGuard uses a similarity threshold of **78%** — scores below this are not considered a match.

| Query image | Top match | Score | Correct? |
|-------------|-----------|-------|----------|
| Phulkari dupatta (close-up) | Phulkari | 83.3% | ✅ |
| Chanderi saree (close-up) | Chanderi | 82.6% | ✅ |
| Ajrakh print fabric | Banarasi | 81.8% | ⚠️ Wrong craft, right category |
| Western leather handbag | No match | — | ✅ |
| Plain blue Western dress | No match | — | ✅ |
| Birkenstock sandal | Kolhapuri | 88.1% | ❌ Shape false positive |

---

##  Known Limitations

### 1. CLIP was not trained on Indian crafts
CLIP was trained on 400 million general internet images — not on Indian craft-specific data. It understands broad visual categories ("patterned textile", "flat sandal") but cannot reliably distinguish between visually similar crafts such as Ajrakh vs Banarasi, or Ikat vs Phulkari. These craft categories share colour palettes, geometric patterns, and textile structures that CLIP treats as equivalent.

### 2. Shape matching causes false positives for footwear
A Birkenstock sandal scores 88.1% similarity to Kolhapuri chappals — not because it copies the craft, but because both are flat open sandals of similar colour. CLIP matches overall visual shape and tone, not craft-specific pattern elements.

### 3. Whole-image matching
CLIP analyses the entire uploaded image. A full runway photo (model + background + outfit) produces less reliable results than a close-up product shot. The model matches the overall scene rather than isolating the specific product.

### 4. Small dataset
655 images across 8 categories (~82 per category) is sufficient for a prototype but too small for reliable fine-grained classification.

---

##  Future Work

| Improvement | What it solves |
|-------------|---------------|
| **Fine-tune CLIP** on craft-specific data | Teach the model the difference between Ajrakh and Banarasi at a pattern level |
| **Add a YOLO pre-filter** | Detect and crop the specific product (sandal, fabric) before passing to CLIP — eliminates background noise |
| **Add a negative class** | Train the model on ~500 generic Western products so it learns what a non-match looks like |
| **Expand the dataset** | More images per category improves embedding quality and reduces inter-category confusion |
| **Publish fine-tuned model on Hugging Face** | Make it the first open-source model for Indian craft classification |

---


Built by Bhavini Kasana · (https://www.linkedin.com/in/bhavini-kasana-0b65151a9/) · (bhavini.kasana@edu.esiee.fr)

*This is a portfolio project built to demonstrate applied ML for social impact.*
