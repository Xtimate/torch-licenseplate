import os
import sys

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from api.database import (
    add_to_watchlist,
    check_watchlist,
    get_watchlist,
    remove_from_watchlist,
)

router = APIRouter()


class WatchlistEntry(BaseModel):
    text: str
    notes: str | None = None


@router.post("/watchlist")
def add_watchlist(entry: WatchlistEntry):
    add_to_watchlist(entry.text.upper(), entry.notes)
    return {"status": "ok", "text": entry.text.upper()}


@router.delete("/watchlist/{text}")
def remove_watchlist(text: str):
    remove_from_watchlist(text.upper())
    return {"status": "ok"}


@router.get("/watchlist")
def list_watchlist():
    return get_watchlist()


@router.get("/watchlist/check/{text}")
def chec(text: str):
    result = check_watchlist(text.upper())
    if result:
        return {"match": True, "entry": result}
    return {"match": False}
