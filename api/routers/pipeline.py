import hashlib
import io
import os
import sys

from fastapi import APIRouter, File, Request, UploadFile
from PIL import Image
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.database import check_watchlist, insert_plate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import pipeline

router = APIRouter()

limiter = Limiter(key_func=get_remote_address)


@router.post("/pipeline")
@limiter.limit("10/minute")
async def pipeline_endpoint(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image_hash = hashlib.md5(contents).hexdigest()

    cached = request.app.state.cache.get(image_hash)
    if cached is not None:
        return cached

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result = pipeline.run_pipeline(  # type: ignore
        request.app.state.detector,
        request.app.state.recognizer,
        image,
        request.app.state.device,
        request.app.state.temperature,
    )
    for plate in result:
        insert_plate(
            text=plate["text"],
            country=plate.get("country"),
            confidence=plate.get("confidence", 0.0),
            valid_format=plate.get("valid_format", False),
            source="pipeline",
        )
        watch = check_watchlist(plate["text"])
        if watch:
            plate["watchlist_hit"] = True
            plate["watchlist_notes"] = watch.get("notes")

    request.app.state.cache.set(image_hash, result)
    return result


@router.post("/pipeline/batch")
@limiter.limit("10/minute")
async def pipeline_batch_endpoint(
    request: Request, files: list[UploadFile] = File(...)
):
    images = [
        Image.open(io.BytesIO(await file.read())).convert("RGB") for file in files
    ]
    result = pipeline.run_pipeline_batch(  # type: ignore
        request.app.state.detector,
        request.app.state.recognizer,
        images,
        request.app.state.temperature,
    )
    return result
