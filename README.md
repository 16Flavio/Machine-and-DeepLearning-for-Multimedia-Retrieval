# Machine & Deep Learning for Multimedia Retrieval

Academic project (FPMs, MA1 Q2) exploring content-based multimedia retrieval with
both classical and deep-learning approaches, served through a Flask web app.

## Overview

Two retrieval engines are exposed via a web UI:

1. **Unimodal image retrieval (Cars dataset)** — query an image and get visually
   similar cars back. Supported descriptors:
   - Classical: **ORB**, **SIFT** (Bag-of-Visual-Words, 256 clusters), **HOG**
   - Deep learning: **ConvNeXt-Base** and **DINOv2 ViT-B/14**, in both
     classification-head and metric-learning variants
   - Metrics: Euclidienne, Cosinus, Chi carré, Bhattacharyya

2. **Multimodal retrieval (Flickr8k dataset)** — text-to-image and
   image-to-image search using **CLIP ViT-B/32** embeddings indexed with
   **FAISS** (cosine similarity over L2-normalized vectors).

## Project layout

```
├── app.py                         Flask entry point (unimodal + multimodal API)
├── Site_Internet/                 Static frontend
│   ├── index.html                 Multimodal page (CLIP + FAISS)
│   └── index2.html                Unimodal page (classical + DL descriptors)
├── src/
│   ├── part1_unimodal/
│   │   ├── search.py              Loads precomputed gallery features & searches
│   │   └── extract_classical.py   Computes HOG / ORB-BoVW / SIFT-BoVW features
│   └── part2_multimodal/
│       ├── clip_encoder.py        CLIP wrapper (images + text, L2-normalized)
│       ├── faiss_index.py         FAISS IndexFlatIP cosine search
│       └── extract_clip.py        One-shot Flickr8k gallery encoder
├── notebooks/                     Data analysis, descriptor tests, evaluation
├── data/raw/                      Cars/ and Flickr8k_dataset/ (git-ignored)
├── models/                        Trained .pth checkpoints (git-ignored)
├── results/                       Evaluation tables, plots, feature banks
├── docs/                          Report, user manual
├── pyproject.toml, uv.lock        Python deps (managed with uv)
└── Dockerfile
```

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Required datasets (place under `data/raw/`):
- `Cars/` — car images used by the unimodal engine
- `Flickr8k_dataset/Images/` — images used by the multimodal engine

## Building the feature banks

The web app serves results from precomputed feature files under
`results/features/`. Generate them once:

```bash
# Classical descriptors (HOG + ORB-BoVW + SIFT-BoVW) over the Cars gallery
python src/part1_unimodal/extract_classical.py

# CLIP ViT-B/32 embeddings over Flickr8k (uses CUDA if available)
python src/part2_multimodal/extract_clip.py
```

Deep-learning gallery features (ConvNeXt / DINOv2) are generated from the
notebook `notebooks/01_descriptor_tests_DeepLearning.ipynb`.

## Running the web app

```bash
python app.py
```

Then open http://localhost:5000 and switch between the two tabs:
- **Moteur de recherche multi modal** — text or image query over Flickr8k
- **Moteur de recherche d'image uni modal à deux niveaux** — image query over Cars

## API

| Method | Path                       | Purpose                                             |
|--------|----------------------------|-----------------------------------------------------|
| GET    | `/api/images`              | List Cars gallery filenames (autocomplete)          |
| GET    | `/cars_dataset/<name>`     | Serve a Cars image                                  |
| GET    | `/flickr_dataset/<name>`   | Serve a Flickr8k image                              |
| POST   | `/api/search`              | Unimodal search `{filename, descriptor, metric, top_k}` |
| POST   | `/api/search_multimodal`   | Multimodal search — JSON `{text, top_k}` or multipart `image + top_k` |

## Authors

Canfyn Marie, Drogo Flavio — FPMs, 2026.
