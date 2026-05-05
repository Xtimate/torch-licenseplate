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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
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
    min_char_confidence: float = 0.0


NL_PATTERNS = [
    r"^\d{2}[A-Z]{3}\d$",  # DD-LLL-D
    r"^[A-Z]{2}\d{3}[A-Z]$",  # LL-DDD-L
    r"^[A-Z]\d{3}[A-Z]{2}$",  # L-DDD-LL
    r"^\d{2}[A-Z]{2}\d{2}$",  # DD-LL-DD
    r"^[A-Z]{2}\d{2}[A-Z]{2}$",  # LL-DD-LL
    r"^\d{2}[A-Z]{2}[A-Z]{2}$",  # DD-LL-LL
]
DE_PATTERNS = [
    r"^[A-Z]{3}\d{4}$",  # LLL-DD-DD
    r"^[A-Z]{2}\d{5}$",  # LL-DDD-DD
    r"^[A-Z]{4}\d{4}$",  # LLLL-DD-DD
]
FR_PATTERNS = [
    r"^[A-Z]{2}\d{3}[A-Z]{2}$"  # LL-DDD-LL
]

BE_PATTERNS = [
    r"^\d[A-Z]{3}\d{3}$",  # 1-ABC-234
]

IT_PATTERNS = [
    r"^[A-Z]{2}\d{3}[A-Z]{2}$",  # AB-123-CD
]

PL_PATTERNS = [
    r"^[A-Z]{2}\d{5}$",  # WA-12345
    r"^[A-Z]{2}\d{4}[A-Z]$",  # WA-1234-B
    r"^[A-Z]{3}\d{4}$",  # WAR-1234
    r"^[A-Z]{3}\d{3}[A-Z]$",  # WAR-123-B
]

SE_PATTERNS = [
    r"^[A-Z]{3}\d{3}$",  # ABC-123
]

ES_PATTERNS = [
    r"^\d{4}[A-Z]{3}$",  # 1234-ABC
]

CONFUSIONS = {
    "0": "D",
    "D": "0",
    "1": "7",
    "7": "1",
    "8": "B",
    "B": "8",
    "5": "S",
    "S": "5",
    "2": "Z",
    "Z": "2",
    "6": "G",
    "G": "6",
}


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


def _try_confusion_fix(text: str) -> tuple[str, str] | tuple[None, None]:
    for i, ch in enumerate(text):
        if ch in CONFUSIONS:
            candidate = text[:i] + CONFUSIONS[ch] + text[i + 1 :]
            valid, country = validate_format(candidate)
            if valid:
                return candidate, country  # type: ignore
    return None, None


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _greedy_ctc(logits: np.ndarray, blank, temperature: float = 1.0) -> tuple:
    scaled = logits / temperature
    probs_full = _softmax(scaled)

    scaled_no_blank = scaled.copy()
    scaled_no_blank[:, blank] = -1e9
    probs_no_blank = _softmax(scaled_no_blank)

    chars, confs = [], []
    prev = None
    for t in range(probs_no_blank.shape[0]):
        token = int(np.argmax(probs_full[t]))
        if token != prev and token != blank:
            chars.append(idx_to_char[token])
            confs.append(float(probs_no_blank[t, token]))
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
    img.save("/tmp/debug_crop.jpg")
    tensor = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        log_probs = torch.log_softmax(output, dim=2)  # type: ignore
        return _greedy_ctc(log_probs)  # type: ignore


def load_recognizer_onnx(model_path: str):
    print(f"loading model from {os.path.abspath(model_path)}")
    return ort.InferenceSession(model_path)


def recognize_from_image_onnx(
    image, session, threshold: float = 0.7, temperature: float = 1.0, _retries: int = 0
) -> RecognitionResult:
    img = image.resize((188, 48)).convert("RGB")
    tensor = to_tensor(img).unsqueeze(0).numpy()
    logits = session.run(None, {"input": tensor})[0]
    blank = int(logits.shape[2] - 1)
    text, char_confs = _greedy_ctc(logits[:, 0, :], blank, temperature)

    if not char_confs:
        return RecognitionResult("", 0.0, [], True, "empty_output")

    if len(text) < 5:
        return RecognitionResult(text, 0.0, [], True, "too_short")

    confidence = float(np.mean(char_confs))
    min_char_conf = float(min(char_confs))
    rejected = confidence < threshold or min_char_conf < threshold * 0.3
    reason = None
    if confidence < threshold:
        reason = f"confidence {confidence:.3f} below threshold {threshold}"
    if min_char_conf < threshold * 0.6:
        reason = f"min char confidence {min_char_conf:.3f} below threshold {threshold * 0.6:.3f}"
    valid_format, country = validate_format(text)

    if not valid_format and _retries < 2:
        embedded, _ = find_embedded_pattern(text)
        if embedded:
            print(f"  retrying ({_retries + 1}/2) — '{text}' contains embedded pattern")
            return recognize_from_image_onnx(
                image, session, threshold, temperature, _retries + 1
            )

    if _retries >= 2:
        trimmed = text[:-1]
        valid_trimmed, country_trimmed = validate_format(trimmed)
        if valid_trimmed:
            return RecognitionResult(
                text=trimmed,
                confidence=confidence,
                char_confidences=char_confs[:-1],
                rejected=rejected,
                rejection_reason=reason,
                valid_format=True,
                country=country_trimmed,
                min_char_confidence=min_char_conf,
            )
    if not valid_format:
        fixed, fixed_country = _try_confusion_fix(text)
        if fixed:
            return RecognitionResult(
                text=fixed,
                confidence=confidence,
                char_confidences=char_confs,
                rejected=rejected,
                rejection_reason=reason,
                valid_format=True,
                country=fixed_country,
                min_char_confidence=min_char_conf,
            )
        print(
            f"  conf={confidence:.3f} min_char={min_char_conf:.3f} rejected={rejected} reason={reason}"
        )

    return RecognitionResult(
        text=text,
        confidence=confidence,
        char_confidences=char_confs,
        rejected=rejected,
        rejection_reason=reason,
        valid_format=valid_format,
        country=country,
        min_char_confidence=min_char_conf,
    )


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
    for pattern in IT_PATTERNS:
        if re.match(pattern, text):
            return True, "IT"
    for pattern in BE_PATTERNS:
        if re.match(pattern, text):
            return True, "BE"
    for pattern in ES_PATTERNS:
        if re.match(pattern, text):
            return True, "ES"
    for pattern in PL_PATTERNS:
        if re.match(pattern, text):
            return True, "PL"
    for pattern in SE_PATTERNS:
        if re.match(pattern, text):
            return True, "SE"
    return False, None


def find_embedded_pattern(text: str) -> tuple[bool, str | None]:
    for length in range(len(text) - 1, 4, -1):
        for start in range(len(text) - length + 1):
            substring = text[start : start + length]
            valid, country = validate_format(substring)
            if valid:
                return True, country
    return False, None
