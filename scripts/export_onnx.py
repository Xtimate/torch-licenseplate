import torch

from src.dataset import CHARS
from src.recognizer import LPRNet

model = LPRNet(num_chars=len(CHARS))
model.load_state_dict(torch.load("checkpoints/lprnet_best_charleswright.pth"))
model.eval()

dummy = torch.randn(1, 3, 48, 188)

torch.onnx.export(
    model,
    dummy,
    "onnx/lprnet.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=12,
)

print("Model exported successfully")
