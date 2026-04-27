import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CHARS, LicensePlateDataset
from recognizer import LPRNet

if __name__ == "__main__":
    model = LPRNet(num_chars=len(CHARS))
    dataset = LicensePlateDataset(size=10000, country="NL")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    loss_fn = nn.CTCLoss(blank=len(CHARS) - 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    if os.path.exists("checkpoints/lprnet.pth"):
        model.load_state_dict(torch.load("checkpoints/lprnet.pth"))
        print("Loaded existing checkpoint")

    for epoch in range(50):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            inputs, labels = batch
            labels = torch.stack(labels, dim=1).view(-1)
            optimizer.zero_grad()
            outputs = model(inputs)
            log_probs = torch.log_softmax(outputs, dim=2)
            input_lengths = torch.full(
                (inputs.size(0),), outputs.size(0), dtype=torch.long
            )
            target_lengths = torch.full((inputs.size(0),), 8, dtype=torch.long)
            loss = loss_fn(log_probs, labels, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()  # once per epoch
        print(f"Epoch {epoch + 1} loss: {total_loss / len(dataloader):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/lprnet.pth")
    print("Model saved to checkpoints/lprnet.pth")

    model.eval()
    with torch.no_grad():
        sample_img, sample_label = dataset[0]
        output = model(sample_img.unsqueeze(0))
        output = torch.log_softmax(output, dim=2)
        pred = output.argmax(dim=2).squeeze(1).tolist()
        print("Predicted:", pred)
        print("Expected:", sample_label)
