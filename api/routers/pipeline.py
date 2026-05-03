import io
import os
import sys

from fastapi import APIRouter, File, Request, UploadFile
from PIL import Image
from slowapi import Limiter
from slowapi.util import get_remote_address

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import pipeline

router = APIRouter()

limiter = Limiter(key_func=get_remote_address)


@router.post("/pipeline")
@limiter.limit("10/minute")
async def pipeline_endpoint(request: Request, file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    result = pipeline.run_pipeline(  # type: ignore
        request.app.state.detector,
        request.app.state.recognizer,
        image,
        request.app.state.device,
    )
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
    )
    return result
