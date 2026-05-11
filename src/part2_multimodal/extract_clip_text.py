from pathlib import Path
import json
import time
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from part2_multimodal.clip_encoder import _ensure_loaded, _unwrap, _normalize
import torch

ROOT = Path(__file__).resolve().parents[2]
CAPTIONS_PATH = ROOT / "data" / "raw" / "Flickr8k_dataset" / "captions.txt"
OUT_DIR = ROOT / "results" / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GALLERY_PATH = OUT_DIR / "CLIP_flickr_text_gallery.npy"
META_PATH = OUT_DIR / "CLIP_flickr_captions.json"

BATCH_SIZE = 256


def _load_pairs():
    pairs = []
    with open(CAPTIONS_PATH, encoding="utf-8") as f:
        for i, raw in enumerate(f):
            if i == 0 and raw.lower().startswith("image,"):
                continue
            line = raw.rstrip("\n")
            if not line:
                continue
            img, _, cap = line.partition(",")
            img = img.strip()
            cap = cap.strip()
            if img and cap:
                pairs.append((img, cap))
    return pairs


@torch.no_grad()
def main():
    pairs = _load_pairs()
    print(f"Captions: {len(pairs)}", flush=True)

    model, processor, device = _ensure_loaded()
    print(f"CLIP loaded on {device}", flush=True)

    t0 = time.time()
    feats_chunks = []
    captions = [c for _, c in pairs]
    images = [i for i, _ in pairs]

    for start in range(0, len(captions), BATCH_SIZE):
        chunk = captions[start:start + BATCH_SIZE]
        inputs = processor(text=chunk, return_tensors="pt", padding=True, truncation=True).to(device)
        out = _unwrap(model.get_text_features(**inputs)).detach().cpu().numpy().astype(np.float32)
        feats_chunks.append(out)
        done = start + len(chunk)
        if done % (BATCH_SIZE * 4) == 0 or done == len(captions):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  encoded {done}/{len(captions)}  ({elapsed:.1f}s, {rate:.1f} cap/s)", flush=True)

    feats = _normalize(np.concatenate(feats_chunks, axis=0))
    np.save(GALLERY_PATH, feats)
    META_PATH.write_text(
        json.dumps({"images": images, "captions": captions}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved gallery: {feats.shape} -> {GALLERY_PATH}", flush=True)
    print(f"Saved metadata: {META_PATH}", flush=True)
    print(f"Done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
