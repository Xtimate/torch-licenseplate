import hashlib
import io
import os
import sys

from fastapi import APIRouter, File, Request, UploadFile
from PIL import Image
from slowapi import Limiter
from slowapi.util import get_remote_address

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import detector  # type: ignore

router = APIRouter()

limiter = Limiter(key_func=get_remote_address)


@router.post("/detect")
@limiter.limit("20/minute")
async def detect(request: Request, file: UploadFile = File(...)):
    image_hash = hashlib.md5(await file.read()).hexdigest()
    cashed = request.state.cache.get(image_hash)
    if cashed is not None:
        return cashed

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    detections = detector.detect_from_image(request.app.state.detector, image)
    request.state.cache.set(image_hash, detections)
    return detections
