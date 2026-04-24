# Machine & Deep Learning for Multimedia Retrieval

Academic project (FPMs, MA1 Q2, course **I-ILIA-014**) exploring
content-based multimedia retrieval with both classical and deep-learning
approaches, served through a containerized Flask web app.

> **Authors** — Canfyn Marie & Drogo Flavio, *Group 05*, 2025-2026.

---

## Overview

Two retrieval engines are exposed via a single web UI:

1. **Unimodal image retrieval — *Cars* dataset** (10 000 images,
   10 brands / 96 brand+model classes). Query an image and retrieve the
   visually closest cars. Available descriptors:
   - **Classical** : ORB-BoVW, SIFT-BoVW (K=256 visual words), HOG
   - **Deep learning** : ConvNeXt-Base and DINOv2 ViT-B/14, in both
     classification-head and metric-learning (SubCenter-ArcFace) variants
   - **Distances** : Euclidean, Cosine (correlation), χ², Bhattacharyya
   - **Multi-descriptor fusion** via Reciprocal Rank Fusion (RRF, c=60)

2. **Multimodal retrieval — *Flickr8K* dataset** (8 000 images, 5
   captions each). Text-to-image and image-to-image search with
   **CLIP ViT-B/32** embeddings, indexed by **FAISS**
   (`IndexFlatIP` over L2-normalized vectors).

A 3D UMAP projection of any descriptor space is also available from the
UI for qualitative inspection.

For a step-by-step usage guide, see
[`docs/user_manual/user_manual.md`](docs/user_manual/user_manual.md).

---

## Project layout

```
├── app.py                           Flask entry point (REST API + static frontend)
├── Dockerfile                       Production image (python:3.11-slim + uv)
├── Site_Internet/                   Static frontend
│   ├── index.html                   Multimodal page (CLIP + FAISS)
│   └── index2.html                  Unimodal page (classical + DL descriptors)
├── src/
│   ├── part1_unimodal/
│   │   ├── search.py                Pre-indexed gallery search (RRF fusion)
│   │   ├── extractors.py            On-demand feature extraction (uploads)
│   │   └── extract_classical.py     Offline HOG / ORB-BoVW / SIFT-BoVW indexing
│   └── part2_multimodal/
│       ├── clip_encoder.py          CLIP wrapper (image + text, L2-normalized)
│       ├── faiss_index.py           FAISS IndexFlatIP cosine search
│       └── extract_clip.py          One-shot Flickr8k gallery encoder
├── notebooks/                       Data analysis & descriptor evaluation
├── data/raw/                        Cars/ and Flickr8k_dataset/ (git-ignored)
├── models/                          Trained .pth checkpoints (git-ignored)
├── results/features/                Pre-indexed gallery features (git-ignored)
├── docs/
│   ├── report/                      LaTeX project report
│   └── user_manual/                 End-user manual
├── pyproject.toml, uv.lock          Python deps (managed with uv)
└── README.md
```

---

## Pre-trained weights & feature banks (Hugging Face)

The trained checkpoints (~1 GB) and pre-computed feature banks
(~500 MB) are **not stored in this Git repository**. They are
distributed through a dedicated Hugging Face model repository:

> **Hugging Face repo** :
> [`https://huggingface.co/16Flavio/mir-cars-flickr8k`](https://huggingface.co/16Flavio/mir-cars-flickr8k)

### What the HF repo contains

| Path on HF | Local destination |
|---|---|
| `models/ConvNeXtBase_*_dim*.pth`           | `models/`           |
| `models/DINOv2_ViTB14_*_dim*.pth`          | `models/`           |
| `features/HOG_*.npy`                       | `results/features/` |
| `features/ORB_bovw256_*.{npy,pkl}`         | `results/features/` |
| `features/SIFT_bovw256_*.{npy,pkl}`        | `results/features/` |
| `features/ConvNeXtBase_*_dim2048_*.npy`    | `results/features/` |
| `features/DINOv2_ViTB14_*_dim2048_*.npy`   | `results/features/` |
| `features/CLIP_flickr_gallery.npy`         | `results/features/` |
| `features/CLIP_flickr_filenames.json`      | `results/features/` |

### Download

Using the `huggingface_hub` CLI (recommended):

```bash
pip install huggingface_hub
huggingface-cli download 16Flavio/mir-cars-flickr8k \
  --local-dir ./hf_cache --local-dir-use-symlinks False

# move artefacts to their expected locations
mkdir -p models results/features
mv hf_cache/models/*.pth         models/
mv hf_cache/features/*           results/features/
rm -rf hf_cache
```

Or programmatically from Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="16Flavio/mir-cars-flickr8k",
    local_dir="hf_cache",
    local_dir_use_symlinks=False,
)
```

---

## Datasets

The two image corpora are **not** redistributed; download them
separately and place them under `data/raw/` :

- **Cars** (10 000 images, used by the unimodal engine) :
  [https://nextcloud.ig.umons.ac.be/s/BooirG6KkHJ58XB](https://nextcloud.ig.umons.ac.be/s/BooirG6KkHJ58XB)
- **Flickr8k** (8 000 images + captions, used by the multimodal engine) :
  [https://github.com/goodwillyoga/Flickr8k_dataset](https://github.com/goodwillyoga/Flickr8k_dataset)

Final structure:

```
data/raw/
├── Cars/                           *.jpg images (e.g. 0_1_BMW_X3_207.jpg)
└── Flickr8k_dataset/
    ├── Images/*.jpg
    └── captions.txt
```

---

## Quick start (Docker)

```bash
git clone https://github.com/16Flavio/Machine-and-DeepLearning-for-Multimedia-Retrieval.git
cd Machine-and-DeepLearning-for-Multimedia-Retrieval

# 1. Pull pre-trained weights & feature banks from Hugging Face
huggingface-cli download 16Flavio/mir-cars-flickr8k \
  --local-dir ./hf_cache --local-dir-use-symlinks False
mkdir -p models results/features
mv hf_cache/models/* models/ && mv hf_cache/features/* results/features/
rm -rf hf_cache

# 2. Place the Cars and Flickr8k images under data/raw/ (see above)

# 3. Build & run
docker build -t mir-app .
docker run -d -p 5000:5000 --name mir mir-app
```

Open [http://localhost:5000](http://localhost:5000).

---

## Local development setup

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
python app.py
```

### Re-building the feature banks from scratch (optional)

If you want to regenerate the descriptors instead of downloading them
from Hugging Face:

```bash
# Classical descriptors over Cars (HOG + ORB-BoVW + SIFT-BoVW)
python src/part1_unimodal/extract_classical.py

# CLIP ViT-B/32 embeddings over Flickr8k (CUDA recommended)
python src/part2_multimodal/extract_clip.py

# Deep-learning gallery features (ConvNeXt / DINOv2)
jupyter notebook notebooks/01_descriptor_tests_DeepLearning.ipynb
```

---

## REST API

| Method | Path                       | Body                               | Purpose |
|--------|----------------------------|-------------------------------------|---------|
| GET    | `/api/images`              | —                                   | List Cars gallery filenames (autocomplete) |
| GET    | `/cars_dataset/<name>`     | —                                   | Serve a Cars image |
| GET    | `/flickr_dataset/<name>`   | —                                   | Serve a Flickr8k image |
| POST   | `/api/search`              | JSON `{filename, descriptors, metric, top_k}` *or* multipart `image + descriptors[] + metric + top_k` | Unimodal search |
| POST   | `/api/search_multimodal`   | JSON `{text, top_k}` *or* multipart `image + top_k` | Multimodal CLIP search |
| GET    | `/api/projection_3d`       | `?descriptor=ConvNextClassifier`    | UMAP 3D projection of a descriptor space |

Full request/response examples are documented in
[`docs/user_manual/user_manual.md`](docs/user_manual/user_manual.md#6-utilisation-programmatique-de-lapi).

---

## Reproducing the report results

The report (`docs/report/main.pdf`) is fully reproducible from this
repository :

| Section                  | Notebook / script                                              |
|--------------------------|----------------------------------------------------------------|
| Section 2 (unimodal)     | `notebooks/01_descriptor_tests_classique.ipynb`                |
|                          | `notebooks/01_descriptor_tests_DeepLearning.ipynb`             |
| Section 3 (multimodal)   | `notebooks/02_clip_exploration.ipynb`                          |
| Dataset analysis         | `notebooks/00_data_analysis.ipynb`                             |
| Aggregate tables         | `results/all_results.csv`, `results/table3_*.csv`, `results/table4_*.csv` |

---

## License & credits

This work was carried out as part of the **I-ILIA-014** course (FPMs,
UMons, 2025-2026) under the supervision of Pr. Mahmoudi, A. Cools,
M. Benkedadra and M. Gloesener.

Pre-trained models : ConvNeXt-Base & DINOv2 (Meta AI),
CLIP ViT-B/32 (OpenAI).
Libraries : PyTorch, timm, transformers, FAISS, OpenCV, scikit-image,
scikit-learn, UMAP, Flask.
