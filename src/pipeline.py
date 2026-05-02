from PIL import Image

from detector import detect_from_image
from recognizer import recognize_from_image_onnx


def run_pipeline(detector, recognizer, image: Image.Image, device):
    detections = detect_from_image(detector, image)
    results = []
    for det in detections:
        crop = image.crop((det["x1"], det["y1"], det["x2"], det["y2"]))
        result = recognize_from_image_onnx(
            crop,
            recognizer,
        )

        if result.rejected or not result.text:
            continue

        results.append(
            {
                "text": result.text,
                "confidence": result.confidence,
                "valid_format": result.valid_format,
                "country": result.country,
                "x1": det["x1"],
                "y1": det["y1"],
                "x2": det["x2"],
                "y2": det["y2"],
                "conf": det["conf"],
            }
        )
    return results


def run_pipeline_batch(detector, recognizer, images: list[Image.Image]):
    results = []
    for i, image in enumerate(images):
        plates = run_pipeline(detector, recognizer, image, None)
        results.append({"image_index": i, "plates": plates})
    return results
