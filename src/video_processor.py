import re

import cv2
import numpy as np
from PIL import Image

from detector import detect_from_image
from recognizer import is_duplicate, recognize_from_image_onnx


def process_video(video_path: str, detector, recognizer, conf: float = 0.3):
    cap = cv2.VideoCapture(video_path)
    seen_plates = set()
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = detect_from_image(detector, image, conf)

        for det in detections:
            crop = image.crop((det["x1"], det["y1"], det["x2"], det["y2"]))
            result = recognize_from_image_onnx(crop, recognizer)

            if result.rejected or not result.text:
                continue
            if not is_duplicate(result.text, seen_plates):
                seen_plates.add(result.text)
                results.append(
                    {
                        "text": result.text,
                        "valid_format": result.valid_format,
                        "country": result.country,
                        "x1": det["x1"],
                        "y1": det["y1"],
                        "x2": det["x2"],
                        "y2": det["y2"],
                        "conf": det["conf"],
                    }
                )
    cap.release()
    return results
