import io
import os
import sys

from fastapi import APIRouter, File, Request, UploadFile
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import pipeline

router = APIRouter()


@router.post("/pipeline")
async def pipeline_endpoint(request: Request, file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    result = pipeline.run_pipeline(
        request.app.state.detector,
        request.app.state.recognizer,
        image,
        request.app.state.device,
    )
    return result
