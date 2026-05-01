import io
import os
import sys
import tempfile

from fastapi import APIRouter, File, Request, UploadFile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import video_processor as video
from api.routers.recognize import router

router = APIRouter()


@router.post("/video")
async def video_endpoint(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
        temp.write(contents)
        tmp_path = temp.name

    results = video.process_video(
        tmp_path,
        request.app.state.detector,
        request.app.state.recognizer,
        request.app.state.conf,
    )

    os.unlink(tmp_path)
    return results
