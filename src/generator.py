import os
import random
import string
import sys

sys.path.insert(0, os.path.dirname(__file__))
import albumentations as A
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BACKGROUNDS = [
    (255, 220, 0),  # NL yellow
    (255, 255, 255),  # white
]

# Characters used by the recognizer (must match CHARS in recognizer.py)
DIGITS = string.digits
LETTERS = string.ascii_uppercase
ALL_CHARS = DIGITS + LETTERS  # no dashes in random mode

# --- Plate format definitions ---
# Each format is a string where D=digit, L=letter, -=literal dash
NL_FORMATS = [
    "DD-LLL-D",  # current NL (e.g. 12-ABC-3)
    "LL-DDD-L",
    "L-DDD-LL",
    "DD-LL-DD",
    "LL-DD-LL",
]

DE_FORMATS = [
    "LLL-DD-DD",  # German style (city code + digits)
    "LL-DDD-DD",
    "LLLL-DD-DD",
]

FR_FORMATS = [
    "LL-DDD-LL",  # French (AB-123-CD)
]

ALL_FORMATS = NL_FORMATS + DE_FORMATS + FR_FORMATS


def _fill_format(fmt):
    """Turn a format string like 'DD-LLL-D' into a plate string like '47-XKR-9'."""
    result = []
    for ch in fmt:
        if ch == "D":
            result.append(random.choice(DIGITS))
        elif ch == "L":
            result.append(random.choice(LETTERS))
        else:
            result.append(ch)  # literal dash
    return "".join(result)


def _random_plate_text(min_len=5, max_len=8):
    """Fully random alphanumeric string, no dashes, variable length."""
    length = random.randint(min_len, max_len)
    return "".join(random.choices(ALL_CHARS, k=length))


def generate_plate(text, country="NL"):
    bg_color = random.choice(BACKGROUNDS)
    img = Image.new("RGB", (520, 110), color=bg_color)
    font = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Bold.ttf", 52)
    font_small = ImageFont.truetype(
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf", 16
    )
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = 40 + (480 - text_width) // 2 - bbox[0]
    y = (110 - text_height) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=(0, 0, 0))

    sbbox = draw.textbbox((0, 0), country, font=font_small)
    stext_width = sbbox[2] - sbbox[0]
    stext_height = sbbox[3] - sbbox[1]
    xs = (40 - stext_width) // 2 - sbbox[0]
    ys = (110 - stext_height) // 2 - sbbox[1] + 20
    draw.text((xs, ys), country, font=font_small, fill=(0, 0, 50))

    draw.rectangle((0, 0, 519, 109), outline=(0, 0, 0), width=3)
    return img


def random_plate(country=None):
    """
    Generate a random plate with mixed strategy:
      - 70% realistic format (NL/DE/FR patterns)
      - 30% fully random string (forces digit variety at all positions)

    Returns (PIL.Image, text_without_dashes)
    """
    roll = random.random()

    if roll < 0.70:
        # Realistic format
        fmt = random.choice(ALL_FORMATS)
        # Pick country label to match format origin
        if fmt in NL_FORMATS:
            label = "NL"
        elif fmt in DE_FORMATS:
            label = "DE"
        else:
            label = "FR"
        text = _fill_format(fmt)
    else:
        # Fully random
        label = random.choice(["NL", "DE", "FR", "BE", "PL"])
        text = _random_plate_text(min_len=5, max_len=8)

    if country is not None:
        label = country

    img = generate_plate(text, label)
    img = augment_plate(img)

    # Strip dashes — return only the chars the model needs to predict
    clean_text = text.replace("-", "")
    return img, clean_text


# --- Augmentation pipeline ---
transform = A.Compose(
    [
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.GaussNoise(p=0.4),
        A.MotionBlur(blur_limit=5, p=0.3),
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.4),  # JPEG artifacts
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.RandomRain(p=0.1),
        A.Perspective(scale=(0.02, 0.05), p=0.3),  # mild keystone distortion
    ]
)


def augment_plate(img):
    img_np = np.array(img)
    transformed = transform(image=img_np)
    return Image.fromarray(transformed["image"])


# --- Quick visual test ---
if __name__ == "__main__":
    os.makedirs("sample_plates", exist_ok=True)
    counts = {"realistic": 0, "random": 0}
    for i in range(20):
        img, text = random_plate()
        img.save(f"sample_plates/{i:02d}_{text}.png")
        # Rough heuristic: random plates have no dashes in the source text
        print(f"  [{i:02d}] {text}")
    print("\nSaved 20 sample plates to ./sample_plates/")
