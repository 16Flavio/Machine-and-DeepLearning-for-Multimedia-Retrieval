from pathlib import Path
import json
import re
from typing import List, Set

import numpy as np
import faiss
from PIL import Image

from .clip_encoder import encode_text, encode_image

ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "results" / "features"
FLICKR_DIR = ROOT / "data" / "raw" / "Flickr8k_dataset" / "Images"
CAPTIONS_PATH = ROOT / "data" / "raw" / "Flickr8k_dataset" / "captions.txt"

GALLERY_PATH = FEATURES_DIR / "CLIP_flickr_gallery.npy"
FILENAMES_PATH = FEATURES_DIR / "CLIP_flickr_filenames.json"
TEXT_GALLERY_PATH = FEATURES_DIR / "CLIP_flickr_text_gallery.npy"
TEXT_META_PATH = FEATURES_DIR / "CLIP_flickr_captions.json"

_state = {"index": None, "filenames": None, "gallery": None}
_text_state = {"index": None, "captions": None, "images": None, "gallery": None}
_captions_state: dict = {}


def _ensure_index():
    if _state["index"] is not None:
        return _state["index"], _state["filenames"]
    if not GALLERY_PATH.exists() or not FILENAMES_PATH.exists():
        raise FileNotFoundError(
            f"CLIP gallery not found. Run src/part2_multimodal/extract_clip.py to build it. "
            f"Expected: {GALLERY_PATH} and {FILENAMES_PATH}"
        )
    gallery = np.load(GALLERY_PATH).astype(np.float32)
    if not gallery.flags.c_contiguous:
        gallery = np.ascontiguousarray(gallery)
    filenames = json.loads(FILENAMES_PATH.read_text(encoding="utf-8"))
    index = faiss.IndexFlatIP(gallery.shape[1])
    index.add(gallery)
    _state["index"] = index
    _state["filenames"] = filenames
    _state["gallery"] = gallery
    return index, filenames


def _top_k(query_vec: np.ndarray, k: int) -> List[dict]:
    index, filenames = _ensure_index()
    q = query_vec.reshape(1, -1).astype(np.float32)
    if not q.flags.c_contiguous:
        q = np.ascontiguousarray(q)
    k = max(1, min(int(k), index.ntotal))
    scores, idx = index.search(q, k)
    return [
        {"filename": filenames[int(i)], "score": float(s)}
        for s, i in zip(scores[0], idx[0])
    ]


def _ensure_text_index():
    if _text_state["index"] is not None:
        return _text_state["index"], _text_state["captions"], _text_state["images"]
    if not TEXT_GALLERY_PATH.exists() or not TEXT_META_PATH.exists():
        raise FileNotFoundError(
            f"CLIP text gallery not found. Run src/part2_multimodal/extract_clip_text.py to build it. "
            f"Expected: {TEXT_GALLERY_PATH} and {TEXT_META_PATH}"
        )
    gallery = np.load(TEXT_GALLERY_PATH).astype(np.float32)
    if not gallery.flags.c_contiguous:
        gallery = np.ascontiguousarray(gallery)
    meta = json.loads(TEXT_META_PATH.read_text(encoding="utf-8"))
    captions = meta["captions"]
    images = meta["images"]
    index = faiss.IndexFlatIP(gallery.shape[1])
    index.add(gallery)
    _text_state["index"] = index
    _text_state["captions"] = captions
    _text_state["images"] = images
    _text_state["gallery"] = gallery
    return index, captions, images


def _top_k_text(query_vec: np.ndarray, k: int) -> List[dict]:
    index, captions, images = _ensure_text_index()
    q = query_vec.reshape(1, -1).astype(np.float32)
    if not q.flags.c_contiguous:
        q = np.ascontiguousarray(q)
    k = max(1, min(int(k), index.ntotal))
    scores, idx = index.search(q, k)
    return [
        {"caption": captions[int(i)], "image": images[int(i)], "score": float(s)}
        for s, i in zip(scores[0], idx[0])
    ]

_NORMALIZE_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_STOPWORDS = {
    "a", "an", "the", "of", "on", "in", "at", "to", "and", "or", "is", "are",
    "with", "for", "by", "from", "into", "as", "it", "this", "that", "his",
    "her", "their", "its", "be", "been", "was", "were",
}


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = _NORMALIZE_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _content_tokens(text: str) -> set:
    return {t for t in _normalize(text).split() if t and t not in _STOPWORDS}


def _load_captions():
    if _captions_state:
        return _captions_state
    image_to_caps: dict = {}
    if CAPTIONS_PATH.exists():
        with open(CAPTIONS_PATH, encoding="utf-8") as f:
            for i, raw in enumerate(f):
                if i == 0 and raw.lower().startswith("image,"):
                    continue
                line = raw.rstrip("\n")
                if not line:
                    continue
                img, _, cap = line.partition(",")
                img = img.strip()
                if not img or not cap:
                    continue
                image_to_caps.setdefault(img, []).append(_normalize(cap))
    _captions_state["image_to_caps"] = image_to_caps
    return _captions_state


def _relevant_for_text(text: str) -> Set[str]:
    state = _load_captions()
    image_to_caps = state["image_to_caps"]
    q_tokens = _content_tokens(text)
    if not q_tokens:
        return set()
    relevant = set()
    for name, caps in image_to_caps.items():
        for c in caps:
            cap_tokens = set(c.split())
            if q_tokens.issubset(cap_tokens):
                relevant.add(name)
                break
    return relevant


def _relevant_for_image(filename: str) -> Set[str]:
    state = _load_captions()
    image_to_caps = state["image_to_caps"]
    query_caps = set(image_to_caps.get(filename, []))
    if not query_caps:
        return set()
    relevant = set()
    for name, caps in image_to_caps.items():
        if any(c in query_caps for c in caps):
            relevant.add(name)
    return relevant


def _pr_curve_topk(topk_filenames: List[str], relevant_set: Set[str]) -> dict:
    total_rel = len(relevant_set)
    if total_rel == 0 or not topk_filenames:
        return {"recall": [], "precision": [], "average_precision": 0.0, "total_relevant": total_rel}
    rel = np.array([1 if f in relevant_set else 0 for f in topk_filenames], dtype=np.int32)
    cum_rel = np.cumsum(rel)
    ranks = np.arange(1, len(rel) + 1, dtype=np.float32)
    precision = cum_rel / ranks
    recall = cum_rel / total_rel
    ap = float(precision.mean())
    return {
        "recall": [0.0] + recall.tolist(),
        "precision": [1.0] + precision.tolist(),
        "average_precision": ap,
        "total_relevant": total_rel,
    }


def list_flickr_filenames() -> List[str]:
    _, filenames = _ensure_index()
    return list(filenames)


_unique_captions_cache: List[str] = []


def list_flickr_captions(query: str = "", limit: int = 20) -> List[str]:
    global _unique_captions_cache
    if not _unique_captions_cache:
        _, captions, _ = _ensure_text_index()
        seen = set()
        unique = []
        for c in captions:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        _unique_captions_cache = unique
    tokens = [t for t in (query or "").lower().split() if t]
    if not tokens:
        return _unique_captions_cache[:limit]
    out = []
    for c in _unique_captions_cache:
        cl = c.lower()
        if all(t in cl for t in tokens):
            out.append(c)
            if len(out) >= limit:
                break
    return out


def search_text(text: str, top_k: int) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("Requête texte vide.")
    results = _top_k(encode_text(text), top_k)
    relevant = _relevant_for_text(text)
    pr = _pr_curve_topk([r["filename"] for r in results], relevant)
    return {"results": results, "pr_curve": pr, "kind": "image"}


def search_image(pil_image: Image.Image, top_k: int, uploaded_name: str | None = None) -> dict:
    if pil_image is None:
        raise ValueError("Aucune image fournie.")
    _, filenames = _ensure_index()
    results = _top_k(encode_image(pil_image), top_k)
    if uploaded_name and uploaded_name in set(filenames):
        relevant = _relevant_for_image(uploaded_name)
    else:
        relevant = set()
    pr = _pr_curve_topk([r["filename"] for r in results], relevant)
    return {"results": results, "pr_curve": pr, "kind": "image"}


def search_image_to_text(pil_image: Image.Image, top_k: int, uploaded_name: str | None = None) -> dict:
    if pil_image is None:
        raise ValueError("Aucune image fournie.")
    index, captions, images = _ensure_text_index()
    q = encode_image(pil_image).reshape(1, -1).astype(np.float32)
    if not q.flags.c_contiguous:
        q = np.ascontiguousarray(q)
    k = max(1, min(int(top_k), index.ntotal))
    scores, idx = index.search(q, k)
    ranked = [int(i) for i in idx[0]]
    results = [
        {"caption": captions[i], "image": images[i], "score": float(s)}
        for s, i in zip(scores[0], ranked)
    ]

    if uploaded_name and uploaded_name in set(images):
        relevant_idx = {i for i, name in enumerate(images) if name == uploaded_name}
        rel = np.array([1 if j in relevant_idx else 0 for j in ranked], dtype=np.int32)
        cum_rel = np.cumsum(rel)
        ranks = np.arange(1, len(rel) + 1, dtype=np.float32)
        precision = cum_rel / ranks
        recall = cum_rel / len(relevant_idx)
        pr = {
            "recall": [0.0] + recall.tolist(),
            "precision": [1.0] + precision.tolist(),
            "average_precision": float(precision.mean()),
            "total_relevant": int(len(relevant_idx)),
        }
    else:
        pr = {"recall": [], "precision": [], "average_precision": 0.0, "total_relevant": 0}

    return {"results": results, "pr_curve": pr, "kind": "text"}
