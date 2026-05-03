import hashlib
import os
import sys

from torch.optim.optimizer import R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import OrderedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import dataset  # type: ignore
import detector as det_module  # type: ignore
import recognizer as rec_module  # type: ignore
from api.config import CONF_THRESHOLD, DETECTOR_WEIGHTS, DEVICE, RECOGNIZER_WEIGHTS
from api.routers import detect, pipeline, recognize, video, webcam
from src.detector import load_detector_onnx
from src.recognizer import load_recognizer_onnx

limiter = Limiter(key_func=get_remote_address)


class LRUCache:
    def __init__(self, maxsize=256):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = load_detector_onnx("onnx/detector_best.onnx")
    app.state.recognizer = load_recognizer_onnx("onnx/lprnet.onnx")
    app.state.device = DEVICE
    app.state.conf = CONF_THRESHOLD
    app.state.cache = LRUCache(maxsize=256)
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(detect.router)
app.include_router(recognize.router)
app.include_router(pipeline.router)
app.include_router(video.router)
app.include_router(webcam.router)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://xtimate.github.io"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
def health(request: Request):
    detector_ok = request.app.state.detector is not None
    recognizer_ok = request.app.state.recognizer is not None

    components = {
        "detector": "ok" if detector_ok else "not loaded",
        "recognizer": "ok" if recognizer_ok else "not loaded",
    }

    all_ok = all([detector_ok, recognizer_ok])

    return JSONResponse(
        content={
            "status": "ok" if all_ok else "degraded",
            "components": components,
        },
        status_code=200 if all_ok else 503,
    )
