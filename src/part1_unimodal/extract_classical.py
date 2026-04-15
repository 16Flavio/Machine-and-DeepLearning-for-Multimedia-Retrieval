from pathlib import Path
import pickle
import time
import numpy as np
import cv2
from PIL import Image
from skimage.feature import hog
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[2]
CARS_DIR = ROOT / "data" / "raw" / "Cars"
OUT_DIR = ROOT / "results" / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUERY_FILENAMES = [
    "0_1_BMW_X3_207.jpg", "0_0_BMW_Serie3Berline_74.jpg", "0_2_BMW_i8_299.jpg",
    "2_0_Volkswagen_Touareg_2822.jpg", "2_4_Volkswagen_Polo_3463.jpg", "2_9_Volkswagen_T-Roc_4209.jpg",
    "4_2_Opel_vivarofourgon_5999.jpg", "4_4_Opel_Insignatourer_6353.jpg", "4_9_Opel_zafiralife_6887.jpg",
    "6_0_Hyundai_Nexo_8282.jpg", "6_3_Hyundai_i10_8837.jpg", "6_5_Hyundai_i30_9125.jpg",
    "8_1_Ford_Puma_11276.jpg", "8_5_Ford_Explorer_11897.jpg", "8_6_Ford_Focus_11951.jpg",
]

IMG_SIZE = 256
VOCAB_SIZE = 256
KMEANS_SAMPLE = 200_000
ORB_NFEATURES = 500
SIFT_NFEATURES = 500


def label_of(filename: str) -> str:
    parts = Path(filename).stem.split("_")
    return f"{parts[0]}_{parts[1]}"


def load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.uint8)


def hog_vec(gray: np.ndarray) -> np.ndarray:
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def orb_desc(gray: np.ndarray, orb):
    kps, desc = orb.detectAndCompute(gray, None)
    if desc is None:
        return np.zeros((0, 32), dtype=np.uint8)
    return desc  # uint8, (n, 32)


def sift_desc(gray: np.ndarray, sift):
    kps, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        return np.zeros((0, 128), dtype=np.float32)
    return desc.astype(np.float32)


def bovw_histogram(desc: np.ndarray, kmeans: MiniBatchKMeans) -> np.ndarray:
    hist = np.zeros(kmeans.n_clusters, dtype=np.float32)
    if len(desc) == 0:
        return hist
    words = kmeans.predict(desc.astype(np.float32))
    np.add.at(hist, words, 1.0)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def build_file_lists():
    queries = set(QUERY_FILENAMES)
    gallery = [p for p in sorted(CARS_DIR.glob("*.jpg")) if p.name not in queries]
    query_paths = [CARS_DIR / q for q in QUERY_FILENAMES]
    return gallery, query_paths


def save_set(tag: str, g_feats, q_feats, g_labels, q_labels):
    np.save(OUT_DIR / f"{tag}_gallery.npy", g_feats)
    np.save(OUT_DIR / f"{tag}_query.npy", q_feats)
    np.save(OUT_DIR / f"{tag}_gallery_labels.npy", g_labels)
    np.save(OUT_DIR / f"{tag}_query_labels.npy", q_labels)
    print(f"  saved {tag}: gallery={g_feats.shape}, query={q_feats.shape}")


def main():
    gallery_paths, query_paths = build_file_lists()
    all_paths = gallery_paths + query_paths
    n_gallery = len(gallery_paths)
    n_query = len(query_paths)
    print(f"Gallery: {n_gallery}, Query: {n_query}")

    g_labels = np.array([label_of(p.name) for p in gallery_paths])
    q_labels = np.array([label_of(p.name) for p in query_paths])

    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)

    hog_all = []
    orb_all = []
    sift_all = []

    t0 = time.time()
    for i, p in enumerate(all_paths):
        gray = load_gray(p)
        hog_all.append(hog_vec(gray))
        orb_all.append(orb_desc(gray, orb))
        sift_all.append(sift_desc(gray, sift))
        if (i + 1) % 500 == 0 or i == len(all_paths) - 1:
            elapsed = time.time() - t0
            print(f"  extraction {i+1}/{len(all_paths)}  ({elapsed:.1f}s)")

    hog_all = np.stack(hog_all).astype(np.float32)
    print(f"HOG shape: {hog_all.shape}")

    print("Fitting ORB KMeans...")
    orb_sample = np.concatenate(orb_all[:n_gallery], axis=0) if orb_all[:n_gallery] else np.zeros((0, 32), dtype=np.uint8)
    if len(orb_sample) > KMEANS_SAMPLE:
        idx = np.random.default_rng(0).choice(len(orb_sample), KMEANS_SAMPLE, replace=False)
        orb_sample = orb_sample[idx]
    print(f"  ORB sample for KMeans: {orb_sample.shape}")
    orb_km = MiniBatchKMeans(n_clusters=VOCAB_SIZE, random_state=0, batch_size=4096, n_init=3)
    orb_km.fit(orb_sample.astype(np.float32))

    print("Fitting SIFT KMeans...")
    sift_sample = np.concatenate(sift_all[:n_gallery], axis=0) if sift_all[:n_gallery] else np.zeros((0, 128), dtype=np.float32)
    if len(sift_sample) > KMEANS_SAMPLE:
        idx = np.random.default_rng(0).choice(len(sift_sample), KMEANS_SAMPLE, replace=False)
        sift_sample = sift_sample[idx]
    print(f"  SIFT sample for KMeans: {sift_sample.shape}")
    sift_km = MiniBatchKMeans(n_clusters=VOCAB_SIZE, random_state=0, batch_size=4096, n_init=3)
    sift_km.fit(sift_sample)

    print("Building BoVW histograms...")
    orb_feats = np.stack([bovw_histogram(d, orb_km) for d in orb_all])
    sift_feats = np.stack([bovw_histogram(d, sift_km) for d in sift_all])

    with open(OUT_DIR / f"ORB_bovw{VOCAB_SIZE}_kmeans.pkl", "wb") as f:
        pickle.dump(orb_km, f)
    with open(OUT_DIR / f"SIFT_bovw{VOCAB_SIZE}_kmeans.pkl", "wb") as f:
        pickle.dump(sift_km, f)

    save_set(
        "HOG",
        hog_all[:n_gallery], hog_all[n_gallery:],
        g_labels, q_labels,
    )
    save_set(
        f"ORB_bovw{VOCAB_SIZE}",
        orb_feats[:n_gallery], orb_feats[n_gallery:],
        g_labels, q_labels,
    )
    save_set(
        f"SIFT_bovw{VOCAB_SIZE}",
        sift_feats[:n_gallery], sift_feats[n_gallery:],
        g_labels, q_labels,
    )
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
