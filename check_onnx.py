import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, "src")
from dataset import CHARS
from recognizer import LPRNet

dummy = torch.randn(1, 3, 48, 188)

# PyTorch
model = LPRNet(num_chars=len(CHARS))
model.load_state_dict(torch.load("checkpoints/lprnet_best.pth", weights_only=True))
model.eval()
with torch.no_grad():
    pt_out = model(dummy)
print("pytorch output shape:", pt_out.shape)
print("pytorch output sample:", pt_out[0, 0, :5])

# ONNX
sess = ort.InferenceSession("onnx/lprnet.onnx")
onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
print("onnx output shape:", onnx_out.shape)
print("onnx output sample:", onnx_out[0, 0, :5])
