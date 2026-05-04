import argparse
import os
import sys
import time

import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, "src")

from dataset import CHARS, idx_to_char
from recognizer import _softmax, load_recognizer_onnx

BLANK = len(CHARS) - 1


def _greedy_ctc_with_logprobs(logits: np.ndarray, temperature: float) -> tuple:
    scaled = logits / temperature
    scaled_no_blank = scaled.copy()
    scaled_no_blank[:, BLANK] = -1e9

    probs_full = _softmax(scaled)
    probs_no_blank = _softmax(scaled_no_blank)
    chars = []
    log_confs = []
    prev = None

    for t in range(probs_full.shape[0]):
        token = int(np.argmax(probs_full[t]))
        if token != prev and token != BLANK:
            chars.append(idx_to_char[token])

            log_confs.append(float(np.log(max(probs_no_blank[t, token], 1e-9))))
        prev = token

    return "".join(chars), log_confs


def _nll(log_confs: list[float]) -> float:
    if not log_confs:
        return 0.0
    return -float(np.mean(log_confs))


def load_plates(data_dir: str) -> list[tuple]:
    mixed_index = os.path.join(data_dir, "index.txt")

    if os.path.exists(mixed_index):
        entries = []
        with open(mixed_index) as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("real_"):
                    continue
                filename, label = line.split(" ", 1)
                img_path = os.path.join(data_dir, filename)
                if not os.path.exists(img_path):
                    continue
                img = Image.open(img_path).convert("RGB").resize((188, 48))
                tensor = to_tensor(img).unsqueeze(0).numpy()
                entries.append((tensor, label))
        print(f"Loaded {len(entries)} real augmented plates from {mixed_index}")
        return entries

    print(
        f"No index.txt found in {data_dir}, generating augmented plates on the fly..."
    )
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
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
            for _ in range(augment_n):
                aug = augment_plate(img).resize((188, 48))
                tensor = to_tensor(aug).unsqueeze(0).numpy()
                entries.append((tensor, label))

    print(f"Generated {len(entries)} augmented plates on the fly")
    return entries


def evaluate_temperature(session, entries: list, temperature: float) -> float:
    n_bins = 10
    bin_correct = np.zeros(n_bins)
    bin_total = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)

    for tensor, label in entries:
        logits = session.run(None, {"input": tensor})[0]
        logits_2d = logits[:, 0, :]
        predicted, log_confs = _greedy_ctc_with_logprobs(logits_2d, temperature)

        if not log_confs:
            continue

        conf = float(np.mean(np.exp(log_confs)))
        correct = predicted == label

        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bin_correct[bin_idx] += int(correct)
        bin_total[bin_idx] += 1
        bin_conf[bin_idx] += conf

    ece = 0.0
    for i in range(n_bins):
        if bin_total[i] == 0:
            continue
        avg_acc = bin_correct[i] / bin_total[i]
        avg_conf = bin_conf[i] / bin_total[i]
        print(
            f"  bin {i}: total={int(bin_total[i])} conf={avg_conf:.2f} acc={avg_acc:.2f}"
        )
        ece += (bin_total[i] / len(entries)) * abs(avg_conf - avg_acc)

    return float(ece)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="onnx/lprnet.onnx")
    parser.add_argument("--data", default="data/mixed")
    parser.add_argument("--t-min", type=float, default=0.5)
    parser.add_argument("--t-max", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=26)
    args = parser.parse_args()

    session = load_recognizer_onnx(args.model)
    entries = load_plates(args.data)
    print(f"Starting temperature search over {len(entries)} entries...")

    temperatures = np.linspace(args.t_min, args.t_max, args.steps)

    best_t = 1.0
    best_nll = float("inf")

    print(f"\n{'T':>6}  {'ECE':>8}  {'time':>6}")
    print("-" * 26)
    for t in temperatures:
        t_start = time.time()
        nll = evaluate_temperature(session, entries, t)
        marker = " ←" if nll < best_nll else ""
        print(f"{t:>6.2f}  {nll:>8.4f}  {time.time() - t_start:>5.2f}s{marker}")
        if nll < best_nll:
            best_nll = nll
            best_t = t

    print(f"\nBest temperature: {best_t:.2f}  (NLL: {best_nll:.4f})")
    print(f"Add to .env:\n  TEMPERATURE={best_t:.2f}")
