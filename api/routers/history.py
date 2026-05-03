import os
import sys

from fastapi import APIRouter, Query

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from api.database import get_history, get_stats

router = APIRouter()


@router.get("/history")
def history(
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    country: str = Query(default=None),
    source: str = Query(default=None),
):
    return get_history(limit=limit, offset=offset, country=country, source=source)


@router.get("/stats")
def stats():
    return get_stats()
