from PIL import Image
from ultralytics import YOLO  # type: ignore


def load_detector(weights_path: str):
    model = YOLO(weights_path)
    return model


def detect_from_image(model, image: Image.Image, conf: float = 0.3):
    results = model(image, conf=conf)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    detections = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        detections.append(
            {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "conf": float(results[0].boxes.conf.cpu().numpy()[i]),
            }
        )
    return detections


def load_detector_onnx(model_path: str):
    from ultralytics import YOLO  # type: ignore

    return YOLO(model_path, task="detect")
