import os
import random
import string
import sys

sys.path.insert(0, os.path.dirname(__file__))
import albumentations as A
import numpy as np
from PIL import Image, ImageDraw, ImageFont

backgrounds = [
    (255, 220, 0),
    (255, 255, 255),
]
bg_color = random.choice(backgrounds)


def generate_plate(text, country="NL"):
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

    draw.rectangle((0, 0, 519, 109), outline=(0, 0, 0), width=3)
    img.save("test.png")
    return img


def random_plate(country):
    digits1 = "".join(random.choices(string.digits, k=2))
    letters = "".join(random.choices(string.ascii_uppercase, k=3))
    digit2 = "".join(random.choices(string.digits, k=1))
    text = digits1 + "-" + letters + "-" + digit2
    img = generate_plate(text, country)
    img = augment_plate(img)
    return img, text


transform = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.MotionBlur(p=0.2),
        A.RandomRain(p=0.1),
    ]
)


def augment_plate(img):
    img_np = np.array(img)
    transformed = transform(image=img_np)
    return Image.fromarray(transformed["image"])
