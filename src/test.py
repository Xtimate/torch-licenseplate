import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from dataset import CHARS, LicensePlateDataset, decode, idx_to_char
from recognizer import LPRNet

# ── helpers ──────────────────────────────────────────────────────────────────


def ctc_decode(output):
    """Greedy CTC decode on a [T, 1, num_chars] log-prob tensor."""
    blank = len(CHARS) - 1
    pred = output.argmax(dim=2).squeeze(1).tolist()
    result = []
    prev = None
    for p in pred:
        if p != prev and p != blank:
            result.append(idx_to_char[p])
        prev = p
    return "".join(result)


def load_model(checkpoint, device):
    model = LPRNet(num_chars=len(CHARS)).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def predict_image(model, img, device):
    """Run inference on a PIL image. Returns predicted string."""
    img = img.resize((188, 48)).convert("RGB")
    tensor = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        log_probs = torch.log_softmax(output, dim=2)  # type: ignore
    return ctc_decode(log_probs)


# ── modes ─────────────────────────────────────────────────────────────────────


def test_synthetic(model, device, n=20):
    """Test on freshly generated synthetic plates."""
    print(f"\n── Synthetic test ({n} plates) ──")
    dataset = LicensePlateDataset(size=n)
    correct = 0
    for i in range(n):
        img, label = dataset[i]
        tensor = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            log_probs = torch.log_softmax(output, dim=2)  # type: ignore
        predicted = ctc_decode(log_probs)
        expected = "".join([idx_to_char[c] for c in label])
        match = "✓" if predicted == expected else "✗"
        print(f"  [{match}] predicted: {predicted:<12} expected: {expected}")
        correct += predicted == expected
    print(f"\n  Accuracy: {correct}/{n} ({100 * correct / n:.0f}%)")


def test_real(model, device, paths):
    """
    Test on real plate images.
    Filenames can encode the ground truth: e.g. 99XNB2.jpg or 99-XNB-2.jpg
    If no ground truth can be parsed, just prints the prediction.
    """
    print(f"\n── Real image test ({len(paths)} images) ──")
    correct = 0
    has_gt = 0
    for path in paths:
        img = Image.open(path).convert("RGB")
        predicted = predict_image(model, img, device)

        # Try to extract ground truth from filename (strip dashes)
        stem = os.path.splitext(os.path.basename(path))[0].upper().replace("-", "")
        gt_chars = [c for c in stem if c in CHARS and c not in ("-", "_")]
        gt = "".join(gt_chars) if gt_chars else None

        if gt:
            match = "✓" if predicted == gt else "✗"
            print(
                f"  [{match}] predicted: {predicted:<12} expected: {gt}  ({os.path.basename(path)})"
            )
            correct += predicted == gt
            has_gt += 1
        else:
            print(f"  [?] predicted: {predicted:<12}  ({os.path.basename(path)})")

    if has_gt:
        print(f"\n  Accuracy: {correct}/{has_gt} ({100 * correct / has_gt:.0f}%)")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPRNet test script")
    parser.add_argument(
        "images",
        nargs="*",
        help="Real plate image files to test. If omitted, runs synthetic test.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/lprnet.pth",
        help="Path to model checkpoint (default: checkpoints/lprnet.pth)",
    )
    parser.add_argument(
        "--synthetic",
        type=int,
        default=20,
        metavar="N",
        help="Number of synthetic plates to test when no images are provided (default: 20)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
    print(f"Device : {device}")
    print(f"Checkpoint: {os.path.abspath(args.checkpoint)}")

    model = load_model(args.checkpoint, device)

    if args.images:
        test_real(model, device, args.images)
    else:
        test_synthetic(model, device, n=args.synthetic)
