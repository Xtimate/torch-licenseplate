import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, "src")
from dataset import CHARS, idx_to_char

BLANK = len(CHARS) - 1


def ctc_decode(output):
    pred = output.argmax(axis=2).squeeze(1).tolist()
    result = []
    prev = None
    for p in pred:
        if p != prev and p != BLANK:
            result.append(idx_to_char[p])
        prev = p
    return "".join(result)


def run(image_path):
    # Load image
    img = Image.open(image_path).convert("RGB").resize((188, 48))
    tensor = to_tensor(img).unsqueeze(0).numpy()

    # LPRNet inference
    sess = ort.InferenceSession("onnx/lprnet.onnx")
    output = sess.run(None, {"input": tensor})[0]
    text = ctc_decode(output)
    print(f"Recognized: {text}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test.png"
    run(path)
