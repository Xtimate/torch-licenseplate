import torch

from dataset import CHARS, LicensePlateDataset, decode
from recognizer import LPRNet


def ctc_decode(pred):
    result = []
    prev = None
    blank = len(CHARS) - 1
    for p in pred:
        if p != prev and p != blank:
            result.append(p)
        prev = p
    return result


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LPRNet(num_chars=len(CHARS))
model.load_state_dict(
    torch.load("checkpoints/lprnet.pth", map_location=device, strict=False)
)
model = model.to(device)
model.eval()

dataset = LicensePlateDataset(size=100, country="NL")

with torch.no_grad():
    sample_img, sample_label = dataset[0]
    output = model(sample_img.unsqueeze(0).to(device))
    output = torch.log_softmax(output, dim=2)
    pred = output.argmax(dim=2).squeeze(1).tolist()
    print("Raw pred:", pred)
    print("Predicted:", decode(ctc_decode(pred)))
    print("Expected:", decode(sample_label))
    print("Predicted text:", decode(ctc_decode(pred)))
    print("Expected text:", decode(sample_label))

correct = 0
for i in range(10):
    sample_img, sample_label = dataset[i]
    with torch.no_grad():
        output = model(sample_img.unsqueeze(0).to(device))
        output = torch.log_softmax(output, dim=2)
        pred = output.argmax(dim=2).squeeze(1).tolist()
        predicted = decode(ctc_decode(pred))
        expected = decode(sample_label)
        print(f"Predicted: {predicted} | Expected: {expected}")
        if predicted == expected:
            correct += 1
print(f"Accuracy: {correct}/10")
