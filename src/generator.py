import os
import random
import string
import sys

sys.path.insert(0, os.path.dirname(__file__))
import albumentations as A
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

BACKGROUNDS = [
    (255, 220, 0),  # NL yellow
    (255, 255, 255),  # white
]
DIGITS = string.digits
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
ALL_CHARS = DIGITS + LETTERS

NL_FORMATS = ["DD-LLL-D", "LL-DDD-L", "L-DDD-LL", "DD-LL-DD", "LL-DD-LL"]
DE_FORMATS = ["LLL-DD-DD", "LL-DDD-DD", "LLLL-DD-DD"]
FR_FORMATS = ["LL-DDD-LL"]
ALL_FORMATS = NL_FORMATS + DE_FORMATS + FR_FORMATS


def _fill_format(fmt):
    result = []
    for ch in fmt:
        if ch == "D":
            result.append(random.choice(DIGITS))
        elif ch == "L":
            result.append(random.choice(LETTERS))
        else:
            result.append(ch)
    return "".join(result)


def _random_plate_text(min_len=5, max_len=8):
    length = random.randint(min_len, max_len)
    return "".join(random.choices(ALL_CHARS, k=length))


def _add_lighting_gradient(img):
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    angle = random.uniform(0, 2 * np.pi)
    strength = random.uniform(0.04, 0.12)
    xs = np.linspace(0, 1, w)
    ys = np.linspace(0, 1, h)
    xg, yg = np.meshgrid(xs, ys)
    gradient = np.cos(angle) * xg + np.sin(angle) * yg
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
    gradient = (gradient * 2 - 1) * strength
    arr += gradient[:, :, np.newaxis] * 255
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _add_surface_wear(img):
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, random.uniform(1, 5), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _add_shadow(img):
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    if random.random() < 0.5:
        x0 = random.randint(0, w // 2)
        x1 = random.randint(x0, w)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[:, x0:x1] = random.uniform(0.05, 0.15)
    else:
        y0 = random.randint(0, h // 2)
        y1 = random.randint(y0, h)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[y0:y1, :] = random.uniform(0.05, 0.12)
    arr -= mask[:, :, np.newaxis] * 255
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _add_glare(img):
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    cx = random.randint(w // 4, 3 * w // 4)
    cy = random.randint(h // 4, 3 * h // 4)
    rx = random.randint(20, w // 3)
    ry = random.randint(10, h // 3)
    strength = random.uniform(0.08, 0.20)
    ys, xs = np.ogrid[:h, :w]
    mask = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2
    mask = np.clip(1 - mask, 0, 1) * strength
    arr += mask[:, :, np.newaxis] * 255
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _draw_eu_strip(draw, img_height, strip_width=52):
    """Draw the blue EU strip on the left side of the plate."""
    # Blue background
    draw.rectangle((0, 0, strip_width, img_height), fill=(0, 51, 153))

    # Yellow stars in a circle (simplified — just a ring of dots)
    cx = strip_width // 2
    star_r = 14
    num_stars = 12
    star_size = 2
    for i in range(num_stars):
        angle = 2 * np.pi * i / num_stars - np.pi / 2
        sx = int(cx + star_r * np.cos(angle))
        sy = int(img_height // 2 - 16 + star_r * np.sin(angle))
        draw.ellipse(
            (sx - star_size, sy - star_size, sx + star_size, sy + star_size),
            fill=(255, 220, 0),
        )

    # "NL" text at the bottom of the strip
    try:
        font_tiny = ImageFont.truetype(
            "/usr/share/fonts/liberation/LiberationSans-Bold.ttf", 13
        )
        tbbox = draw.textbbox((0, 0), "NL", font=font_tiny)
        tx = (strip_width - (tbbox[2] - tbbox[0])) // 2
        ty = img_height - 20
        draw.text((tx, ty), "NL", font=font_tiny, fill=(255, 255, 255))
    except Exception:
        pass


def generate_plate(text, country="NL"):
    bg_color = random.choice(BACKGROUNDS)
    bg_color = tuple(max(0, min(255, c + random.randint(-15, 15))) for c in bg_color)

    img = Image.new("RGB", (520, 110), color=bg_color)
    font = ImageFont.truetype(
        "/home/xtimate/Documents/torch-licenseplate/fonts/CharlesWright-Bold.otf", 72
    )
    font_small = ImageFont.truetype(
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf", 16
    )
    draw = ImageDraw.Draw(img)

    # --- EU strip (NL plates only, 80% of the time to add variety) ---
    strip_width = 0
    if country == "NL" and random.random() < 0.80:
        strip_width = 52
        _draw_eu_strip(draw, 110, strip_width)

    # Text area starts after the strip
    text_area_start = strip_width + 5
    text_area_width = 520 - text_area_start

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = text_area_start + (text_area_width - text_width) // 2 - bbox[0]
    y = (110 - text_height) // 2 - bbox[1]

    text_color = tuple(random.randint(0, 30) for _ in range(3))
    draw.text((x, y), text, font=font, fill=text_color)

    # Country label in strip (only if no EU strip drawn)
    if strip_width == 0:
        sbbox = draw.textbbox((0, 0), country, font=font_small)
        xs = (40 - (sbbox[2] - sbbox[0])) // 2 - sbbox[0]
        ys = (110 - (sbbox[3] - sbbox[1])) // 2 - sbbox[1] + 20
        draw.text((xs, ys), country, font=font_small, fill=(0, 0, 50))

    draw.rectangle((0, 0, 519, 109), outline=(0, 0, 0), width=3)

    img = _add_surface_wear(img)
    if random.random() < 0.7:
        img = _add_lighting_gradient(img)
    if random.random() < 0.3:
        img = _add_shadow(img)
    if random.random() < 0.25:
        img = _add_glare(img)

    return img


def random_plate(country=None):
    roll = random.random()
    if roll < 0.70:
        fmt = random.choice(ALL_FORMATS)
        if fmt in NL_FORMATS:
            label = "NL"
        elif fmt in DE_FORMATS:
            label = "DE"
        else:
            label = "FR"
        text = _fill_format(fmt)
    else:
        label = random.choice(["NL", "DE", "FR", "BE", "PL"])
        text = _random_plate_text(min_len=5, max_len=8)

    if country is not None:
        label = country

    img = generate_plate(text, label)
    img = augment_plate(img)
    clean_text = text.replace("-", "")
    return img, clean_text


transform = A.Compose(
    [
        A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.5),
        A.GaussNoise(p=0.3),
        A.MotionBlur(blur_limit=7, p=0.35),
        A.ImageCompression(compression_type="jpeg", quality=(50, 95), p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.25),
        A.RandomRain(p=0.1),
        A.Perspective(scale=(0.03, 0.08), p=0.5),
        A.Rotate(limit=5, p=0.4),
        A.RandomShadow(p=0.15),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(2, 10),
            hole_width_range=(2, 20),
            fill=0,
            p=0.2,
        ),
    ]
)


def augment_plate(img):
    img_np = np.array(img)
    transformed = transform(image=img_np)
    return Image.fromarray(transformed["image"])


if __name__ == "__main__":
    os.makedirs("sample_plates", exist_ok=True)
    for i in range(20):
        img, text = random_plate()
        img.save(f"sample_plates/{i:02d}_{text}.png")
        print(f"  [{i:02d}] {text}")
    print("\nSaved 20 sample plates to ./sample_plates/")
