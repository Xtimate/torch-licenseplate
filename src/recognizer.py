import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_tensor

from dataset import CHARS, idx_to_char

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.insert(0, os.path.dirname(__file__))

BLANK = len(CHARS) - 1


@dataclass
class RecognitionResult:
    text: str
    confidence: float
    char_confidences: list
    rejected: bool
    rejection_reason: Optional[str] = None
    valid_format: bool = False
    country: Optional[str] = None


NL_PATTERNS = [
    r"^\d{2}[A-Z]{3}\d$",  # DD-LLL-D
    r"^[A-Z]{2}\d{3}[A-Z]$",  # LL-DDD-L
    r"^[A-Z]\d{3}[A-Z]{2}$",  # L-DDD-LL
    r"^\d{2}[A-Z]{2}\d{2}$",  # DD-LL-DD
    r"^[A-Z]{2}\d{2}[A-Z]{2}$",  # LL-DD-LL
]
DE_PATTERNS = [
    r"^[A-Z]{3}\d{4}$",  # LLL-DD-DD
    r"^[A-Z]{2}\d{5}$",  # LL-DDD-DD
    r"^[A-Z]{4}\d{4}$",  # LLLL-DD-DD
]
FR_PATTERNS = [
    r"^[A-Z]{2}\d{3}[A-Z]{2}$"  # LL-DDD-LL
]


class LPRNet(nn.Module):
    def __init__(self, num_chars):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Conv2d(256, num_chars, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        x = x.mean(dim=(2))
        x = x.permute(2, 0, 1)
        return x


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _greedy_ctc(probs: np.ndarray, blank: int) -> tuple:
    """probs: softmaxed [T, num_chars]"""
    chars, confs = [], []
    prev = None
    for t in range(probs.shape[0]):
        token = int(np.argmax(probs[t]))
        peak = float(probs[t, token])
        if token != prev or token == BLANK:
            prev = token
            continue
        chars.append(idx_to_char[token])
        confs.append(peak)
        prev = token
    return "".join(chars), confs


def load_recognizer(num_chars, model_path, device):
    model = LPRNet(num_chars)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model


def recognize_from_image(image, model, device):
    img = image.resize((188, 48)).convert("RGB")
    tensor = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        log_probs = torch.log_softmax(output, dim=2)
        return ctc_decode(log_probs)


def load_recognizer_onnx(model_path: str):
    return ort.InferenceSession(model_path)


def recognize_from_image_onnx(
    image, session, threshold: float = 0.7
) -> RecognitionResult:
    img = image.resize((188, 48)).convert("RGB")
    tensor = to_tensor(img).unsqueeze(0).numpy()
    print(
        f"tensor shape: {tensor.shape}, min: {tensor.min():.3f}, max: {tensor.max():.3f}"
    )
    logits = session.run(None, {"input": tensor})[0]  # [T, 1, num_chars]
    print(
        f"logits shape: {logits.shape}, min: {logits.min():.3f}, max: {logits.max():.3f}"
    )
    print(f"raw argmax: {logits.argmax(axis=2).squeeze().tolist()}")
    probs = _softmax(logits[:, 0, :])  # [T, num_chars]
    print(f"top tokens: {np.argmax(probs, axis=1).tolist()}")
    blank = logits.shape[2] - 1  # last token is blank
    text, char_confs = _greedy_ctc(probs, blank)

    if not char_confs:
        return RecognitionResult("", 0.0, [], True, "empty_output")

    confidence = float(np.mean(char_confs))
    rejected = confidence < threshold
    reason = (
        f"confidence {confidence:.3f} below threshold {threshold}" if rejected else None
    )

    return RecognitionResult(text, confidence, char_confs, rejected, reason)


def is_duplicate(text: str, seen: set[str], max_distance: int = 2) -> bool:
    for seen_plate in seen:
        if _levenshtein(text, seen_plate) <= max_distance:
            return True
    return False


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    row = list(range(len(b) + 1))
    for c1 in a:
        new_row = [row[0] + 1]
        for j, c2 in enumerate(b):
            new_row.append(min(row[j + 1] + 1, new_row[-1] + 1, row[j] + (c1 != c2)))
        row = new_row
    return row[-1]


def validate_format(text: str) -> tuple[bool, str | None]:
    for pattern in NL_PATTERNS:
        if re.match(pattern, text):
            return True, "NL"
    for pattern in DE_PATTERNS:
        if re.match(pattern, text):
            return True, "DE"
    for pattern in FR_PATTERNS:
        if re.match(pattern, text):
            return True, "FR"
    return False, None
