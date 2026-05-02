import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from detector import detect_from_image, load_detector

real_augment = A.Compose(
    [  # type: ignore
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.8),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        A.MotionBlur(blur_limit=9, p=0.4),
        A.ImageCompression(compression_type="jpeg", quality_range=(40, 95), p=0.6),
        A.GaussianBlur(blur_limit=5, p=0.3),
        A.Perspective(scale=(0.03, 0.10), p=0.6),
        A.Rotate(limit=6, p=0.5),
        A.RandomShadow(p=0.3),
        A.RandomRain(p=0.15),
        A.CoarseDropout(
            num_holes_range=(1, 5),
            hole_height_range=(2, 12),
            hole_width_range=(2, 25),
            fill=0,
            p=0.25,
        ),
    ]
)

VALID_CHARS = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") - set("IO")


def parse_label(stem):
    clean = stem.upper().replace("_", "").replace(" ", "")
    clean = "".join(c for c in clean if c in VALID_CHARS)
    return clean if clean else None


def get_crop(img, detector):
    detections = detect_from_image(detector, img, conf=0.25)
    if not detections:
        return img
    best = max(detections, key=lambda d: d["conf"])
    x1, y1, x2, y2 = best["x1"], best["y1"], best["x2"], best["y2"]
    pad = 4
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(img.width, x2 + pad), min(img.height, y2 + pad)
    return img.crop((x1, y1, x2, y2))


def process_real_plates(real_dir, detector, out_dir, augment_n):
    entries = []
    image_paths = []

    for country in ("nl", "fr", "de"):
        folder = Path(real_dir) / country
        if not folder.exists():
            print(f"Warning: {folder} not found, skipping.")
            continue
        for p in folder.iterdir():
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                image_paths.append((p, country.upper()))

    print(f"Found {len(image_paths)} real images.")

    for idx, (path, country) in enumerate(
        tqdm(image_paths, desc="Processing real plates")
    ):
        label = parse_label(path.stem)
        if label is None:
            print(f" Skipping {path.name} - couldn't parse label")
            continue

        img = Image.open(path).convert("RGB")
        crop = get_crop(img, detector)

        for aug_idx in range(augment_n):
            aug_np = real_augment(image=np.array(crop))["image"]
            aug_img = Image.fromarray(aug_np)
            filename = f"real_{idx:03d}_{label}_{aug_idx:02d}.jpg"
            aug_img.save(os.path.join(out_dir, filename), "JPEG", quality=92)
            entries.append((filename, label))

    print(f"  → {len(entries)} real augmented crops saved")
    return entries


def _generate_synthetic(args):
    idx, out_dir = args
    from generator import random_plate

    img, text = random_plate()
    filename = f"synth_{idx:06d}_{text}.jpg"
    img.save(os.path.join(out_dir, filename), "JPEG", quality=92)
    return filename, text


def process_synthetic(size, out_dir, num_workers):
    entries = []
    print(f"Generating {size} synthetic plates...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_generate_synthetic, (i, out_dir)): i for i in range(size)
        }
        for future in tqdm(as_completed(futures), total=size):
            filename, text = future.result()
            entries.append((filename, text))
    return entries


def write_index(entries, out_dir):
    entries.sort(key=lambda e: e[0])
    index_path = os.path.join(out_dir, "index.txt")
    with open(index_path, "w") as f:
        for filename, label in entries:
            f.write(f"{filename} {label}\n")
    print(f"Index written to {index_path} ({len(entries)} entries)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=str, required=True)
    parser.add_argument("--detector", type=str, default="onnx/detector_best.onnx")
    parser.add_argument("--synthetic", type=int, default=20000)
    parser.add_argument("--augment", type=int, default=25)
    parser.add_argument("--out", type=str, default="data/mixed")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    detector = load_detector(args.detector)
    real_entries = process_real_plates(args.real, detector, args.out, args.augment)
    synth_entries = process_synthetic(args.synthetic, args.out, args.workers)
    all_entries = real_entries + synth_entries
    write_index(all_entries, args.out)

    print(f"\nDone.")
    print(f"  Real (augmented) : {len(real_entries)}")
    print(f"  Synthetic        : {len(synth_entries)}")
    print(f"  Total            : {len(all_entries)}")
