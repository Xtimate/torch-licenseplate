import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CHARS, LicensePlateDataset, idx_to_char
from recognizer import LPRNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def ctc_decode(output):
    """Greedy CTC decode — collapse repeats and remove blank (last index)."""
    blank = len(CHARS) - 1
    pred = output.argmax(dim=2).squeeze(1).tolist()  # [T]
    result = []
    prev = None
    for p in pred:
        if p != prev and p != blank:
            result.append(idx_to_char[p])
        prev = p
    return "".join(result)


def collate_fn(batch):
    """
    Custom collate to handle variable-length labels.
    Returns:
        imgs:           [B, C, H, W]
        labels_flat:    1-D tensor of all label indices concatenated
        target_lengths: [B] actual label length per sample
    """
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_flat = torch.tensor([idx for l in labels for idx in l], dtype=torch.long)
    return imgs, labels_flat, target_lengths


if __name__ == "__main__":
    model = LPRNet(num_chars=len(CHARS)).to(device)
    dataset = LicensePlateDataset(size=10000)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    loss_fn = nn.CTCLoss(blank=len(CHARS) - 1, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    checkpoint_path = os.path.abspath("checkpoints/lprnet.pth")
    print(checkpoint_path)
    print(os.path.exists(checkpoint_path))
    if os.path.exists(checkpoint_path):
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device), strict=False
        )
        print("Loaded existing checkpoint")

    for epoch in range(50):
        model.train()
        total_loss = 0

        for imgs, labels_flat, target_lengths in tqdm(
            dataloader, desc=f"Epoch {epoch + 1}"
        ):
            imgs = imgs.to(device)
            labels_flat = labels_flat.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)  # [T, B, num_chars]
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
            f"Epoch {epoch + 1} loss: {avg_loss:.4f}  lr: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # --- Per-epoch sample decode (5 plates) ---
        model.eval()
        with torch.no_grad():
            correct = 0
            print("  Samples:")
            for i in range(5):
                img, label = dataset[i]
                out = model(img.unsqueeze(0).to(device))
                out = torch.log_softmax(out, dim=2)
                predicted = ctc_decode(out)
                expected = "".join([idx_to_char[c] for c in label])
                match = "✓" if predicted == expected else "✗"
                print(f"    [{match}] predicted: {predicted:<12} expected: {expected}")
                correct += predicted == expected
            print(f"  Sample accuracy: {correct}/5")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print("Model saved to checkpoints/lprnet.pth")
