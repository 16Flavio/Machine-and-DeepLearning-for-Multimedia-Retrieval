from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "results" / "features"
CARS_DIR = ROOT / "data" / "raw" / "Cars"

QUERY_FILENAMES = [
    "0_1_BMW_X3_207.jpg", "0_0_BMW_Serie3Berline_74.jpg", "0_2_BMW_i8_299.jpg",
    "2_0_Volkswagen_Touareg_2822.jpg", "2_4_Volkswagen_Polo_3463.jpg", "2_9_Volkswagen_T-Roc_4209.jpg",
    "4_2_Opel_vivarofourgon_5999.jpg", "4_4_Opel_Insignatourer_6353.jpg", "4_9_Opel_zafiralife_6887.jpg",
    "6_0_Hyundai_Nexo_8282.jpg", "6_3_Hyundai_i10_8837.jpg", "6_5_Hyundai_i30_9125.jpg",
    "8_1_Ford_Puma_11276.jpg", "8_5_Ford_Explorer_11897.jpg", "8_6_Ford_Focus_11951.jpg",
]

DESCRIPTOR_TAGS = {
    "ConvNextClassifier": "ConvNeXtBase_classifier_dim2048",
    "ConvNextMetric": "ConvNeXtBase_metric_dim2048",
    "VitMetric": "DINOv2_ViTB14_metric_dim2048",
    "HOG": "HOG",
    "ORB": "ORB_bovw256",
    "SIFT": "SIFT_bovw256",
}

SUPPORTED_METRICS = {"Euclidienne", "Cosinus", "Chi carré", "Bhattacharyya"}

_cache: dict = {}


def label_of(filename: str) -> str:
    parts = Path(filename).stem.split("_")
    if len(parts) < 2:
        return ""
    return f"{parts[0]}_{parts[1]}"


def _gallery_filenames() -> list:
    if "files" not in _cache:
        queries = set(QUERY_FILENAMES)
        _cache["files"] = [p.name for p in sorted(CARS_DIR.glob("*.jpg")) if p.name not in queries]
    return _cache["files"]


def _load(tag: str):
    if tag in _cache:
        return _cache[tag]
    g = np.load(FEATURES_DIR / f"{tag}_gallery.npy")
    q = np.load(FEATURES_DIR / f"{tag}_query.npy")
    try:
        g_lab = np.load(FEATURES_DIR / f"{tag}_gallery_labels.npy")
    except FileNotFoundError:
        g_lab = np.array([label_of(f) for f in _gallery_filenames()])
    try:
        q_lab = np.load(FEATURES_DIR / f"{tag}_query_labels.npy")
    except FileNotFoundError:
        q_lab = np.array([label_of(f) for f in QUERY_FILENAMES])
    _cache[tag] = (g, q, g_lab, q_lab)
    return g, q, g_lab, q_lab


def _get_query_vector(filename: str, tag: str):
    gallery, query, g_lab, q_lab = _load(tag)
    if filename in QUERY_FILENAMES:
        idx = QUERY_FILENAMES.index(filename)
        return query[idx], q_lab[idx]
    files = _gallery_filenames()
    if filename in files:
        idx = files.index(filename)
        return gallery[idx], g_lab[idx]
    raise ValueError(f"Image '{filename}' introuvable dans la gallery/queries.")


def _distances(q: np.ndarray, G: np.ndarray, metric: str) -> np.ndarray:
    q = q.astype(np.float32)
    G = G.astype(np.float32)
    if metric == "Euclidienne":
        return np.linalg.norm(G - q, axis=1)
    if metric == "Cosinus":
        qn = q / (np.linalg.norm(q) + 1e-12)
        Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
        return 1.0 - Gn @ qn
    if metric == "Chi carré":
        qa = np.abs(q) + 1e-12
        Ga = np.abs(G) + 1e-12
        return 0.5 * np.sum((Ga - qa) ** 2 / (Ga + qa), axis=1)
    if metric == "Bhattacharyya":
        qp = np.abs(q) + 1e-12
        qp /= qp.sum()
        Gp = np.abs(G) + 1e-12
        Gp /= Gp.sum(axis=1, keepdims=True)
        bc = np.sum(np.sqrt(Gp * qp), axis=1)
        return -np.log(bc + 1e-12)
    raise ValueError(f"Métrique inconnue: {metric}")


def _pr_curve_topk(topk_labels: np.ndarray, query_label, full_gallery_labels: np.ndarray) -> dict:
    total_rel = int((full_gallery_labels == query_label).sum())
    if total_rel == 0:
        return {"recall": [], "precision": [], "average_precision": 0.0}
    rel = (topk_labels == query_label).astype(np.int32)
    k = len(rel)
    cum_rel = np.cumsum(rel)
    ranks = np.arange(1, k + 1, dtype=np.float32)
    precision = cum_rel / ranks
    recall = cum_rel / total_rel
    ap = float(precision.mean()) if k > 0 else 0.0
    return {
        "recall": [0.0] + recall.tolist(),
        "precision": [1.0] + precision.tolist(),
        "average_precision": ap,
        "total_relevant": total_rel,
    }


RRF_CONST = 60


def _rank_vector(order: np.ndarray, n: int) -> np.ndarray:
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    return rank


def _run_multi(q_vecs_by_tag: dict, metric: str, top_k: int, query_label_by_tag: dict):
    tags = list(q_vecs_by_tag.keys())
    first_tag = tags[0]

    _, _, first_g_lab, _ = _load(first_tag)
    n = len(first_g_lab)

    fused_score = np.zeros(n, dtype=np.float64)
    for tag in tags:
        gallery, _, _, _ = _load(tag)
        dists = _distances(q_vecs_by_tag[tag], gallery, metric)
        order = np.argsort(dists)
        rank = _rank_vector(order, n)
        fused_score += 1.0 / (RRF_CONST + rank + 1)

    fused_order = np.argsort(-fused_score)
    k = max(1, min(int(top_k), n))
    top_idx = fused_order[:k]

    files = _gallery_filenames()
    results = [{"filename": files[i], "score": float(fused_score[i])} for i in top_idx]

    q_label = query_label_by_tag.get(first_tag)
    if q_label is not None:
        pr = _pr_curve_topk(first_g_lab[top_idx], q_label, first_g_lab)
    else:
        pr = {"recall": [], "precision": [], "average_precision": 0.0}
    return results, pr


def _coerce_descriptors(descriptors) -> list:
    if isinstance(descriptors, str):
        descriptors = [descriptors]
    descriptors = [d for d in (descriptors or []) if d]
    if not descriptors:
        raise ValueError("Aucun descripteur sélectionné.")
    unknown = [d for d in descriptors if d not in DESCRIPTOR_TAGS]
    if unknown:
        raise ValueError(
            "Descripteur(s) non disponible(s): " + ", ".join(unknown) +
            ". Supportés: " + ", ".join(DESCRIPTOR_TAGS) + "."
        )
    seen = set()
    out = []
    for d in descriptors:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def search(filename: str, descriptors, metric: str, top_k: int) -> dict:
    descriptors = _coerce_descriptors(descriptors)
    if metric not in SUPPORTED_METRICS:
        raise ValueError("Métrique '" + str(metric) + "' non supportée.")
    if not filename:
        raise ValueError("Aucune image de requête fournie.")

    q_vecs, q_labels = {}, {}
    for d in descriptors:
        tag = DESCRIPTOR_TAGS[d]
        q_vec, q_label = _get_query_vector(filename, tag)
        q_vecs[tag] = q_vec
        q_labels[tag] = q_label

    results, pr = _run_multi(q_vecs, metric, top_k, q_labels)
    first_tag = DESCRIPTOR_TAGS[descriptors[0]]
    return {
        "results": results,
        "pr_curve": pr,
        "query_label": str(q_labels[first_tag]),
        "descriptors": descriptors,
    }


def search_uploaded(image, descriptors, metric: str, top_k: int, query_label: str | None = None) -> dict:
    descriptors = _coerce_descriptors(descriptors)
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Métrique '{metric}' non supportée.")

    from .extractors import extract
    q_vecs, q_labels = {}, {}
    for d in descriptors:
        tag = DESCRIPTOR_TAGS[d]
        q_vecs[tag] = extract(d, image)
        q_labels[tag] = _normalize_query_label(query_label, tag) if query_label else None

    results, pr = _run_multi(q_vecs, metric, top_k, q_labels)
    return {
        "results": results,
        "pr_curve": pr,
        "query_label": query_label or "",
        "descriptors": descriptors,
    }


def _normalize_query_label(query_label_str: str, tag: str):
    _, _, g_lab, q_lab = _load(tag)
    if g_lab.dtype.kind in ("U", "S", "O"):
        return query_label_str
    for i, name in enumerate(QUERY_FILENAMES):
        if label_of(name) == query_label_str:
            return q_lab[i].item() if hasattr(q_lab[i], "item") else q_lab[i]
    for i, name in enumerate(_gallery_filenames()):
        if label_of(name) == query_label_str:
            return g_lab[i].item() if hasattr(g_lab[i], "item") else g_lab[i]
    return None
