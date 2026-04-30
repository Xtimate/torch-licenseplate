import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from dataset import CHARS, LicensePlateDataset, char_to_idx, idx_to_char
from recognizer import LPRNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

REAL_PLATES_DIR = "data/real-plates/eu-license-plates/eu"
AUGMENT_TIMES = 20  # each real plate repeated 20x with augmentation


# ── Real plates dataset ────────────────────────────────────────────────────────

real_aug = A.Compose(
    [
        A.Rotate(limit=5, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.GaussNoise(p=0.5),
        A.RandomBrightnessContrast(p=0.7),
        A.HueSaturationValue(p=0.3),
        A.Perspective(scale=(0.02, 0.05), p=0.4),
        A.Downscale(scale_min=0.5, scale_max=0.9, p=0.3),
        A.Sharpen(p=0.3),
        A.ImageCompression(compression_type="jpeg", quality=(60, 95), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.3),
    ]
)


def encode_real(text):
    """Encode plate text, skipping chars not in CHARS."""
    text = text.upper().replace(" ", "")
    return [char_to_idx[c] for c in text if c in char_to_idx and c not in ("-", "_")]


class RealPlatesDataset(Dataset):
    def __init__(self, root, augment_times=200):
        self.augment_times = augment_times
        self.samples = []

        for country in ["de", "fr", "nl"]:
            folder = os.path.join(root, country)
            if not os.path.exists(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue
                label = os.path.splitext(fname)[0]
                encoded = encode_real(label)
                if len(encoded) < 2:  # skip if too short to be meaningful
                    continue
                self.samples.append((os.path.join(folder, fname), encoded))

        print(
            f"✅ Real plates: {len(self.samples)} unique → {len(self.samples) * augment_times} augmented"
        )

    def __len__(self):
        return len(self.samples) * self.augment_times

    def __getitem__(self, idx):
        path, encoded = self.samples[idx % len(self.samples)]
        img = np.array(Image.open(path).convert("RGB"))

        # Keep one clean copy per image, augment the rest
        if idx >= len(self.samples):
            img = real_aug(image=img)["image"]

        img = Image.fromarray(img).resize((188, 48))
        tensor = to_tensor(img)
        return tensor, encoded


# ── Collate ────────────────────────────────────────────────────────────────────


def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_flat = torch.tensor([idx for l in labels for idx in l], dtype=torch.long)
    return imgs, labels_flat, target_lengths


# ── CTC decode ─────────────────────────────────────────────────────────────────


def ctc_decode(output):
    blank = len(CHARS) - 1
    pred = output.argmax(dim=2).squeeze(1).tolist()
    result, prev = [], None
    for p in pred:
        if p != prev and p != blank:
            result.append(idx_to_char[p])
        prev = p
    return "".join(result)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.abspath("checkpoints/lprnet.pth")
    best_checkpoint_path = os.path.abspath("checkpoints/lprnet_best.pth")
    drive_best = None

    # ── Datasets ───────────────────────────────────────────────────────────────
    synthetic = LicensePlateDataset(size=50000, data_dir="data/plates")
    print(f"Synthetic plates: {len(synthetic)}")

    real = RealPlatesDataset(REAL_PLATES_DIR, augment_times=AUGMENT_TIMES)

    combined = ConcatDataset([synthetic, real])
    print(f"Total combined:   {len(combined)}")

    dataloader = DataLoader(
        combined,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = LPRNet(num_chars=len(CHARS)).to(device)
    loss_fn = nn.CTCLoss(blank=len(CHARS) - 1, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # Load existing checkpoint if available
    if os.path.exists(checkpoint_path):
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device), strict=False
        )
        print("▶ Resumed from existing checkpoint")

    best_loss = float("inf")

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(100):
        model.train()
        total_loss = 0

        for imgs, labels_flat, target_lengths in tqdm(
            dataloader, desc=f"Epoch {epoch + 1}"
        ):
            imgs = imgs.to(device)
            labels_flat = labels_flat.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            log_probs = torch.log_softmax(outputs, dim=2)
            input_lengths = torch.full(
                (imgs.size(0),), outputs.size(0), dtype=torch.long, device=device
            )

            loss = loss_fn(log_probs, labels_flat, input_lengths, target_lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(
            f"Epoch {epoch + 1:03d} | loss: {avg_loss:.4f} | lr: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Save best checkpoint locally + to Drive
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            if drive_best:
                torch.save(model.state_dict(), drive_best)
            print(f"  ★ New best ({best_loss:.4f}) saved locally + to Drive")

        # Per-epoch sample decode on 5 synthetic plates
        model.eval()
        with torch.no_grad():
            correct = 0
            print("  Samples:")
            for i in range(5):
                img, label = synthetic[i]
                out = model(img.unsqueeze(0).to(device))
                out = torch.log_softmax(out, dim=2)
                predicted = ctc_decode(out)
                expected = "".join([idx_to_char[c] for c in label])
                match = "✓" if predicted == expected else "✗"
                print(f"    [{match}] predicted: {predicted:<12} expected: {expected}")
                correct += predicted == expected
            print(f"  Sample accuracy: {correct}/5")

    torch.save(model.state_dict(), checkpoint_path)

    print(f"\nTraining complete.")
    print(f"  Final : {checkpoint_path}")
    print(f"  Best  : {best_checkpoint_path}  (loss {best_loss:.4f})")
