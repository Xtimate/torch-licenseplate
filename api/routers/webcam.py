import io
import os
import sys

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from detector import detect_from_image
from recognizer import is_duplicate, recognize_from_image_onnx

router = APIRouter()


@router.websocket("/webcam")
async def webcam(websocket: WebSocket):
    app = websocket.app
    detector = app.state.detector
    recognizer = app.state.recognizer
    conf = app.state.conf

    await websocket.accept()
    seen_plates = set()
    try:
        while True:
            data = await websocket.receive_bytes()
            image = Image.open(io.BytesIO(data)).convert("RGB")
            detections = detect_from_image(detector, image, conf)

            results = []
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
            await websocket.send_json(results)

    except WebSocketDisconnect:
        pass
