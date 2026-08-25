"""Generate pseudo-thermal counterparts of an RGB anomaly-detection dataset.

Creates a mirrored directory tree (<dst>) with one pseudo-thermal image per
RGB image, preserving exact relative paths/filenames so that the paired
dataset loader can resolve thermal files by mirroring the RGB relative path.

Approximation: grayscale inverted (warm/bright objects emit more), gamma
compression, Gaussian blur (thermal-optics MTF), mild Gaussian noise.
"""
import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def to_pseudo_thermal(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    thermal = 1.0 - gray                      # invert contrast
    thermal = np.power(thermal, 0.8)          # gamma / emissivity approximation
    thermal = cv2.GaussianBlur(thermal, (0, 0), sigmaX=2.0)  # optics MTF
    noise = np.random.normal(0.0, 0.02, thermal.shape).astype(np.float32)
    thermal = np.clip(thermal + noise, 0.0, 1.0)
    return (thermal * 255.0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Pseudo-thermal dataset generator")
    ap.add_argument("--src", type=str, required=True, help="RGB dataset root (e.g., data/MVTec)")
    ap.add_argument("--dst", type=str, required=True, help="Output root (e.g., data/MVTec_T)")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    files = []
    for root, _, names in os.walk(args.src):
        for n in names:
            if os.path.splitext(n)[1].lower() in IMG_EXTS:
                files.append(os.path.join(root, n))

    print(f"Found {len(files)} images under {args.src}")
    skipped = written = 0
    for f in tqdm(files, desc="pseudo-thermal"):
        rel = os.path.relpath(f, args.src)
        out_path = os.path.join(args.dst, rel)
        if os.path.exists(out_path):
            skipped += 1
            continue
        img = cv2.imread(f)
        if img is None:
            print(f"[warn] unreadable: {f}")
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, to_pseudo_thermal(img))
        written += 1

    print(f"Done. written={written} skipped(existing)={skipped}")


if __name__ == "__main__":
    main()
