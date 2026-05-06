import os
import sys

from fastapi import APIRouter, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from api.database import get_analytics, get_history, get_stats

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.get("/history")
@limiter.limit("20/minute")
def history(
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    country: str = Query(default=None),
    source: str = Query(default=None),
):
    return get_history(limit=limit, offset=offset, country=country, source=source)


@router.get("/stats")
@limiter.limit("10/minute")
def stats(request: Request):
    return get_stats()


@router.get("/analytics")
@limiter.limit("10/minute")
def analytics(request: Request):
    return get_analytics()
