import io
import os
import sys

from fastapi import APIRouter, File, Request, UploadFile
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import video_processor  # type: ignore
from api.database import check_watchlist, insert_plate

router = APIRouter()


@router.post("/video")
async def video(request: Request, file: UploadFile = File(...)):
    contents = await file.read()

    results = video_processor.process_video(
        contents,
        request.app.state.detector,
        request.app.state.recognizer,
        conf=request.app.state.conf,
    )

    # Save each unique plate to the database
    for plate in results:
        insert_plate(
            text=plate["text"],
            country=plate.get("country"),
            confidence=plate.get("confidence", 0),
            valid_format=plate.get("valid_format", False),
            source="video",
        )
        # Flag watchlist hits
        watch = check_watchlist(plate["text"])
        if watch:
            plate["watchlist_hit"] = True
            plate["watchlist_notes"] = watch.get("notes")

    return results
