import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextlib import asynccontextmanager

from fastapi import FastAPI

import dataset
import detector as det_module
import recognizer as rec_module
from api.config import CONF_THRESHOLD, DETECTOR_WEIGHTS, DEVICE, RECOGNIZER_WEIGHTS
from api.routers import detect, pipeline, recognize


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = det_module.load_detector(DETECTOR_WEIGHTS)
    app.state.recognizer = rec_module.load_recognizer(
        len(dataset.CHARS), RECOGNIZER_WEIGHTS, DEVICE
    )
    app.state.device = DEVICE
    app.state.conf = CONF_THRESHOLD
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(detect.router)
app.include_router(recognize.router)
app.include_router(pipeline.router)


@app.get("/health")
def health():
    return {"status": "ok"}
