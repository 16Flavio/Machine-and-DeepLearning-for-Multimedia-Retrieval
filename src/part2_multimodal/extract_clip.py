from pathlib import Path
import json
import time
import sys
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from part2_multimodal.clip_encoder import _ensure_loaded, _encode_image_batch, _normalize

ROOT = Path(__file__).resolve().parents[2]
FLICKR_DIR = ROOT / "data" / "raw" / "Flickr8k_dataset" / "Images"
OUT_DIR = ROOT / "results" / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128


def main():
    paths = sorted(FLICKR_DIR.glob("*.jpg"))
    print(f"Flickr8k images: {len(paths)}", flush=True)

    model, processor, device = _ensure_loaded()
    print(f"CLIP loaded on {device}", flush=True)

    feats_chunks = []
    t0 = time.time()
    batch = []
    for i, p in enumerate(paths):
        try:
            batch.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"  skipped {p.name}: {e}", flush=True)
            batch.append(Image.new("RGB", (224, 224)))
        if len(batch) == BATCH_SIZE or i == len(paths) - 1:
            feats_chunks.append(_encode_image_batch(batch, model, processor, device))
            batch = []
            done = i + 1
            if done % (BATCH_SIZE * 4) == 0 or done == len(paths):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  encoded {done}/{len(paths)}  ({elapsed:.1f}s, {rate:.1f} img/s)", flush=True)

    feats = _normalize(np.concatenate(feats_chunks, axis=0))
    filenames = [p.name for p in paths]
    np.save(OUT_DIR / "CLIP_flickr_gallery.npy", feats)
    (OUT_DIR / "CLIP_flickr_filenames.json").write_text(
        json.dumps(filenames), encoding="utf-8"
    )
    print(f"Saved gallery: {feats.shape} -> {OUT_DIR / 'CLIP_flickr_gallery.npy'}", flush=True)
    print(f"Done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
