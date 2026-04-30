from PIL import Image

from detector import detect_from_image
from recognizer import recognize_from_image


def run_pipeline(detector, recognizer, image: Image.Image, device):
    detections = detect_from_image(detector, image)
    results = []
    for det in detections:
        crop = image.crop((det["x1"], det["y1"], det["x2"], det["y2"]))
        text = recognize_from_image(crop, recognizer, device)
        results.append(
            {
                "text": text,
                "x1": det["x1"],
                "y1": det["y1"],
                "x2": det["x2"],
                "y2": det["y2"],
                "conf": det["conf"],
            }
        )
    return results
