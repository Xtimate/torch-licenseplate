import os
import sys

from fastapi import APIRouter, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from api.database import get_heatmap_data, get_locations, get_plate_sightings

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.get("/plates/{text}/history")  # type: ignore
@limiter.limit("30/minute")
def plate_history(text: str, request: Request):
    return get_plate_sightings(text.upper())


@router.get("/heatmap")
@limiter.limit("10/minute")
def heatmap(request: Request):
    return get_heatmap_data()


@router.get("/locations")
@limiter.limit("10/minute")
def locations(request: Request):
    return get_locations()
