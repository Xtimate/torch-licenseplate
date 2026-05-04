import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "src")

from dataset import CHARS, idx_to_char
from recognizer import _softmax, load_recognizer_onnx

BLANK = len(CHARS) - 1


def _greedy_ctc_with_logprobs(logits: np.ndarray, temperature: float) -> tuple:
    scaled = logits / temperature
    probs = _softmax(scaled)

    chars = []
    log_confs = []
    prev = None

    for t in range(probs.shape[0]):
        token = int(np.argmax(probs[t]))

        if token != prev and token != BLANK:
            chars.append(idx_to_char[token])
            log_confs.append(float(max(np.log(probs[t, token]), 1e-9)))
        prev = token

    return "".join(chars), log_confs


def _nll(log_confs: list[float]) -> float:
    if not log_confs:
        return 0.0
    return -float(np.mean(log_confs))


def load_plates(data_dir: str) -> list[tuple]:
    # First check if data/mixed exists and has an index.txt
    mixed_index = os.path.join(data_dir, "index.txt")

    if os.path.exists(mixed_index):
        # Load only real_ entries — synthetic labels are reliable but real ones
        # are what we actually want to calibrate against.
        entries = []
        with open(mixed_index) as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("real_"):
                    continue
                filename, label = line.split(" ", 1)
                img_path = os.path.join(data_dir, filename)
                if os.path.exists(img_path):
                    entries.append((img_path, label))
        print(f"Loaded {len(entries)} real augmented plates from {mixed_index}")
        return entries

    # Fallback — generate augmented real plates on the fly
    print(
        f"No index.txt found in {data_dir}, generating augmented plates on the fly..."
    )
    import numpy as np

    from generator import augment_plate

    raw_dir = "eu-plates/eu-license-plates/eu"
    entries = []
    augment_n = 25

    for country in ("nl", "fr", "de"):
        folder = os.path.join(raw_dir, country)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fname)[0].upper().replace("-", "")
            label = "".join(c for c in stem if c in CHARS and c not in ("-", "_"))
            if not label:
                continue
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert("RGB")
            for _ in range(augment_n):
                # augment_plate applies the same albumentations pipeline used in training
                aug = augment_plate(img)
                entries.append((aug, label))  # aug is a PIL image here, not a path

    print(f"Generated {len(entries)} augmented plates on the fly")
    return entries


def evaluate_temperature(session, entries: list, temperature: float) -> float:
    from torchvision.transforms.functional import to_tensor

    all_nll = []
    for img_or_path, label in entries:
        if isinstance(img_or_path, str):
            img = Image.open(img_or_path).resize((188, 48)).convert("RGB")
        else:
            img = img_or_path.resize((188, 48)).convert("RGB")

        tensor = to_tensor(img).unsqueeze(0).numpy()
        logits = session.run(None, {"input": tensor})[0]  # [T, 1, num_chars]
        logits_2d = logits[:, 0, :]  # [T, num_chars]

        predicted, log_confs = _greedy_ctc_with_logprobs(logits_2d, temperature)

        if predicted == label and log_confs:
            all_nll.append(_nll(log_confs))

    return float(np.mean(all_nll)) if all_nll else float("inf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="onnx/lprnet.onnx")
    parser.add_argument("--data", default="eu-plates/eu-license-plates/eu")
    parser.add_argument("--t-min", type=float, default=0.5)
    parser.add_argument("--t-max", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=26)
    parser.add_argument("--data", default="data/mixed")
    args = parser.parse_args()

    session = load_recognizer_onnx(args.model)
    entries = load_plates(args.data)

    temperatures = np.linspace(args.t_min, args.t_max, args.steps)

    best_t = 1.0
    best_nll = float("inf")

    print(f"\n{'T':>6}  {'NLL':>18}")
    print("-" * 18)
    for t in temperatures:
        nll = evaluate_temperature(session, entries, t)
        marker = " ←" if nll < best_nll else ""
        print(f"{t:>6.2f}  {nll:>8.4f}{marker}")
        if nll < best_nll:
            best_nll = nll
            best_t = t

    print(f"\nBest temperature: {best_t:.2f}  (NLL: {best_nll:.4f})")
    print(f"Add to config.yaml:\n temperature: {best_t:.2f}")
