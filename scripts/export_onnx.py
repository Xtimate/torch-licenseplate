import sys

import torch

sys.path.insert(0, "src")

from dataset import CHARS  # type: ignore
from recognizer import LPRNet  # type: ignore

model = LPRNet(num_chars=len(CHARS))
model.load_state_dict(torch.load("checkpoints/lprnet_best.pth"))
model.eval()

dummy = torch.randn(1, 3, 48, 188)
with torch.no_grad():
    out = model(dummy)
print("output shape", out.shape)
print("output sample", out[0, 0, :5])
torch.onnx.export(
    model,
    dummy,  # type: ignore
    "onnx/lprnet.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=12,
)

print("Model exported successfully")
