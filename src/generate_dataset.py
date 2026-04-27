"""
Pre-generates synthetic license plate images to disk so training
doesn't bottleneck on CPU rendering.

Usage:
    python src/generate_dataset.py --size 20000 --out data/plates
    python src/generate_dataset.py --size 20000 --out data/plates --workers 8

Output structure:
    data/plates/
        000000_13BSRB.jpg
        000001_XK492L.jpg
        ...
        index.txt   # one line per image: "filename label"
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))


def _generate_one(args):
    """Worker function — must be top-level for multiprocessing pickle."""
    idx, out_dir = args
    from generator import random_plate  # import inside worker

    img, text = random_plate()
    filename = f"{idx:06d}_{text}.jpg"
    img.save(os.path.join(out_dir, filename), "JPEG", quality=92)
    return filename, text


def generate_dataset(size, out_dir, num_workers):
    os.makedirs(out_dir, exist_ok=True)

    index_path = os.path.join(out_dir, "index.txt")
    entries = []

    print(f"Generating {size} plates → {out_dir}  (workers={num_workers})")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_generate_one, (i, out_dir)): i for i in range(size)}
        for future in tqdm(as_completed(futures), total=size):
            filename, text = future.result()
            entries.append((filename, text))

    # Sort by index so index.txt is deterministic
    entries.sort(key=lambda e: e[0])
    with open(index_path, "w") as f:
        for filename, text in entries:
            f.write(f"{filename} {text}\n")

    print(f"Done. Index written to {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20000)
    parser.add_argument("--out", type=str, default="data/plates")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    generate_dataset(args.size, args.out, args.workers)
