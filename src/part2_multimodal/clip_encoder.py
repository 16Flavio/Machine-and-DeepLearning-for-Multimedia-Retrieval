from typing import Iterable, List
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

_state = {"model": None, "processor": None, "device": None}


def _ensure_loaded():
    if _state["model"] is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        _state["model"] = model
        _state["processor"] = processor
        _state["device"] = device
    return _state["model"], _state["processor"], _state["device"]


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32)


def _unwrap(out):
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]
    return out


def _encode_image_batch(batch, model, processor, device) -> np.ndarray:
    inputs = processor(images=batch, return_tensors="pt").to(device)
    out = _unwrap(model.get_image_features(**inputs))
    return out.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def encode_images(images: Iterable[Image.Image], batch_size: int = 64) -> np.ndarray:
    model, processor, device = _ensure_loaded()
    feats: List[np.ndarray] = []
    batch: List[Image.Image] = []
    for img in images:
        batch.append(img.convert("RGB"))
        if len(batch) == batch_size:
            feats.append(_encode_image_batch(batch, model, processor, device))
            batch = []
    if batch:
        feats.append(_encode_image_batch(batch, model, processor, device))
    if not feats:
        return np.zeros((0, 512), dtype=np.float32)
    return _normalize(np.concatenate(feats, axis=0))


@torch.no_grad()
def encode_image(path_or_pil) -> np.ndarray:
    img = path_or_pil if isinstance(path_or_pil, Image.Image) else Image.open(path_or_pil)
    return encode_images([img])[0]


@torch.no_grad()
def encode_text(text: str) -> np.ndarray:
    model, processor, device = _ensure_loaded()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    out = _unwrap(model.get_text_features(**inputs)).detach().cpu().numpy().astype(np.float32)
    return _normalize(out)[0]


@torch.no_grad()
def encode_texts(texts: Iterable[str], batch_size: int = 256) -> np.ndarray:
    model, processor, device = _ensure_loaded()
    feats: List[np.ndarray] = []
    batch: List[str] = []
    for t in texts:
        batch.append(t if isinstance(t, str) else str(t))
        if len(batch) == batch_size:
            inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
            out = _unwrap(model.get_text_features(**inputs))
            feats.append(out.detach().cpu().numpy().astype(np.float32))
            batch = []
    if batch:
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
        out = _unwrap(model.get_text_features(**inputs))
        feats.append(out.detach().cpu().numpy().astype(np.float32))
    if not feats:
        return np.zeros((0, 512), dtype=np.float32)
    return _normalize(np.concatenate(feats, axis=0))
