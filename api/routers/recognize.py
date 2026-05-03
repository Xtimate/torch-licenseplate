import io
import os
import sys

from fastapi import APIRouter, File, Request, UploadFile
from PIL import Image
from slowapi import Limiter
from slowapi.util import get_remote_address

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import recognizer  # type: ignore

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/recognize")
@limiter.limit("20/minute")
async def recognize(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result = recognizer.recognize_from_image_onnx(
        image, request.app.state.recognizer, threshold=request.app.state.conf
    )

    if result.rejected:
        return {
            "text": None,
            "confidence": result.confidence,
            "reason": result.rejection_reason,
        }
    return {
        "text": result.text,
        "confidence": result.confidence,
        "valid_format": result.valid_format,
        "country": result.country,
    }
