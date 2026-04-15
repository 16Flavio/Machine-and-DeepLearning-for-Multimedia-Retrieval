from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "results" / "features"
MODELS_DIR = ROOT / "models"

IMG_SIZE_CLASSICAL = 256
VOCAB_SIZE = 256

_cache: dict = {}

def _to_gray(img: Image.Image) -> np.ndarray:
    g = img.convert("L").resize((IMG_SIZE_CLASSICAL, IMG_SIZE_CLASSICAL))
    return np.asarray(g, dtype=np.uint8)


def hog_vector(img: Image.Image) -> np.ndarray:
    from skimage.feature import hog
    return hog(
        _to_gray(img),
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def _load_kmeans(kind: str):
    key = f"kmeans_{kind}"
    if key in _cache:
        return _cache[key]
    path = FEATURES_DIR / f"{kind}_bovw{VOCAB_SIZE}_kmeans.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Vocabulaire {kind} introuvable ({path.name}). "
            "Relancez `python src/part1_unimodal/extract_classical.py` pour régénérer."
        )
    with open(path, "rb") as f:
        km = pickle.load(f)
    _cache[key] = km
    return km


def _bovw(desc: np.ndarray, km) -> np.ndarray:
    hist = np.zeros(km.n_clusters, dtype=np.float32)
    if len(desc) == 0:
        return hist
    words = km.predict(desc.astype(np.float32))
    np.add.at(hist, words, 1.0)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def orb_vector(img: Image.Image) -> np.ndarray:
    gray = _to_gray(img)
    orb = _cache.setdefault("orb", cv2.ORB_create(nfeatures=500))
    _kps, desc = orb.detectAndCompute(gray, None)
    if desc is None:
        desc = np.zeros((0, 32), dtype=np.uint8)
    return _bovw(desc, _load_kmeans("ORB"))


def sift_vector(img: Image.Image) -> np.ndarray:
    gray = _to_gray(img)
    sift = _cache.setdefault("sift", cv2.SIFT_create(nfeatures=500))
    _kps, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        desc = np.zeros((0, 128), dtype=np.float32)
    return _bovw(desc.astype(np.float32), _load_kmeans("SIFT"))

def _torch_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_val_transform(img_size: int):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.10)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _build_convnext(embedding_size: int):
    import timm
    import torch.nn as nn
    import torch
    import torch.nn.functional as F

    class GeM(nn.Module):
        def __init__(self, p=3.0, eps=1e-6):
            super().__init__()
            self.p = nn.Parameter(torch.ones(1) * p)
            self.eps = eps

        def forward(self, x):
            return F.avg_pool2d(
                x.clamp(min=self.eps).pow(self.p),
                (x.size(-2), x.size(-1)),
            ).pow(1.0 / self.p)

    class ConvNeXtExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(
                'convnext_base.fb_in22k_ft_in1k',
                pretrained=False, num_classes=0, global_pool='',
            )
            in_features = self.backbone.num_features
            self.pool = GeM(p=3.0)
            self.bn1 = nn.BatchNorm1d(in_features)
            self.embedding = nn.Linear(in_features, embedding_size)

        def forward(self, x):
            f = self.backbone.forward_features(x)
            f = self.pool(f).flatten(1)
            f = self.bn1(f)
            f = self.embedding(f)
            return F.normalize(f, p=2, dim=1)

    return ConvNeXtExtractor()


def _build_dinov2(embedding_size: int):
    import timm
    import torch.nn as nn
    import torch.nn.functional as F

    class DINOv2Extractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(
                'vit_base_patch14_dinov2.lvd142m',
                pretrained=False, num_classes=0, global_pool='token',
                img_size=224, dynamic_img_size=True,
            )
            in_features = self.backbone.num_features
            self.bn1 = nn.BatchNorm1d(in_features)
            self.embedding = nn.Linear(in_features, embedding_size)

        def forward(self, x):
            f = self.backbone(x)
            f = self.bn1(f)
            f = self.embedding(f)
            return F.normalize(f, p=2, dim=1)

    return DINOv2Extractor()


def _load_dl_model(kind: str, ckpt_name: str, img_size: int, embedding_size: int):
    key = f"dl_{ckpt_name}"
    if key in _cache:
        return _cache[key]

    import torch

    ckpt_path = MODELS_DIR / f"{ckpt_name}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint manquant: {ckpt_path}")

    if kind == "convnext":
        model = _build_convnext(embedding_size)
    elif kind == "dinov2":
        model = _build_dinov2(embedding_size)
    else:
        raise ValueError(kind)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    device = _torch_device()
    model.to(device)

    transform = _build_val_transform(img_size)
    _cache[key] = (model, transform, device)
    return _cache[key]


def _dl_forward(img: Image.Image, kind: str, ckpt_name: str, img_size: int, embedding_size: int) -> np.ndarray:
    import torch
    model, transform, device = _load_dl_model(kind, ckpt_name, img_size, embedding_size)
    x = transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x)
    return feat.squeeze(0).detach().cpu().numpy().astype(np.float32)


def convnext_classifier_vector(img: Image.Image) -> np.ndarray:
    return _dl_forward(img, "convnext", "ConvNeXtBase_classifier_dim2048", img_size=384, embedding_size=2048)


def convnext_metric_vector(img: Image.Image) -> np.ndarray:
    return _dl_forward(img, "convnext", "ConvNeXtBase_metric_dim2048", img_size=384, embedding_size=2048)


def dinov2_metric_vector(img: Image.Image) -> np.ndarray:
    return _dl_forward(img, "dinov2", "DINOv2_ViTB14_metric_dim2048", img_size=224, embedding_size=2048)


EXTRACTORS = {
    "HOG": hog_vector,
    "ORB": orb_vector,
    "SIFT": sift_vector,
    "ConvNextClassifier": convnext_classifier_vector,
    "ConvNextMetric": convnext_metric_vector,
    "VitMetric": dinov2_metric_vector,
}


def extract(descriptor: str, img: Image.Image) -> np.ndarray:
    if descriptor not in EXTRACTORS:
        raise ValueError(f"Descripteur inconnu: {descriptor}")
    return EXTRACTORS[descriptor](img)
