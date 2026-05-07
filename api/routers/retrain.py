import os
import subprocess
import sys

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from api.config import RETRAIN_SECRET

router = APIRouter()


class RetrainRequest(BaseModel):
    secret: str
    epochs: int = 20


@router.post("/retrain")
async def retrain(body: RetrainRequest, request: Request):
    if not RETRAIN_SECRET or body.secret != RETRAIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    subprocess.Popen(
        [sys.executable, "src/retrain_from_queue.py", "--epochs", str(body.epochs)],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )

    return {
        "ok": True,
        "message": f"Retraining started for {body.epochs} epochs. Check server logs for progress.",
    }
