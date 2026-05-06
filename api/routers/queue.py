import base64
import os
import sys

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from api.database import (
    get_queue_stats,
    get_review_item_crop,
    get_review_queue,
    label_review_item,
    reject_review_item,
)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


class LabelRequest(BaseModel):
    text: str


@router.get("/")
@limiter.limit("20/minute")
def queue_list(request: Request, limit: int = 20, offset: int = 0):
    return get_review_queue(limit=limit, offset=offset)


@router.get("/stats")
@limiter.limit("20/minute")
def queue_stats(request: Request):
    return get_queue_stats()


@router.get("/{item_id}/crop")
@limiter.limit("60/minute")
def queue_crop(item_id: int, request: Request):
    crop = get_review_item_crop(item_id)
    if crop is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return Response(content=crop, media_type="image/jpeg")


@router.post("/{item_id}/label")
@limiter.limit("30/minute")
def queue_label(item_id: int, request: Request, body: LabelRequest):
    label_review_item(item_id, body.text)  # type: ignore
    return {"ok": True, "id": item_id, "labeled_text": body.text.upper().strip()}


@router.post("/{item_id}/reject")
@limiter.limit("30/minute")
def queue_reject(item_id: int, request: Request):
    reject_review_item(item_id)
    return {"ok": True, "id": item_id}
