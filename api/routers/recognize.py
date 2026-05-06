import io
import os
import sys

from fastapi import APIRouter, File, Request, UploadFile
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import recognizer  # type: ignore
from api.database import check_watchlist, insert_plate, maybe_queue_for_review
from api.notifier import send_telegram_alert

router = APIRouter()


@router.post("/recognize")
async def recognize(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result = recognizer.recognize_from_image_onnx(
        image,
        request.app.state.recognizer,
        threshold=request.app.state.conf,
        temperature=request.app.state.temperature,
    )

    if result.rejected:
        return {
            "text": None,
            "confidence": result.confidence,
            "reason": result.rejection_reason,
        }

    # Save to database
    insert_plate(
        text=result.text,
        country=result.country,
        confidence=result.confidence,
        valid_format=result.valid_format,
        source="recognize",
    )

    maybe_queue_for_review(
        predicted_text=result.text,
        confidence=result.confidence,
        source="recognize",
        crop_bytes=contents,
    )

    # Check watchlist
    watch = check_watchlist(result.text)
    response = {
        "text": result.text,
        "confidence": result.confidence,
        "char_confidences": result.char_confidences,
        "valid_format": result.valid_format,
        "country": result.country,
    }
    if watch:
        response["watchlist_hit"] = True
        response["watchlist_notes"] = watch.get("notes")
        await send_telegram_alert(
            text=result.text,
            confidence=result.confidence,
            country=result.country,
            notes=watch.get("notes"),
        )

    return response
