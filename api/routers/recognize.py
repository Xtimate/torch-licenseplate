import io
import os
import sys

from fastapi import APIRouter, File, Request, UploadFile
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import recognizer

router = APIRouter()


@router.post("/recognize")
async def recognize(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    text = recognizer.recognize_from_image_onnx(image, request.app.state.recognizer)
    return {"text": text}
