import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import dataset
import detector as det_module
import recognizer as rec_module
from api.config import CONF_THRESHOLD, DETECTOR_WEIGHTS, DEVICE, RECOGNIZER_WEIGHTS
from api.routers import detect, pipeline, recognize, video, webcam
from src.detector import load_detector_onnx
from src.recognizer import load_recognizer_onnx


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = load_detector_onnx("onnx/detector_best.onnx")
    app.state.recognizer = load_recognizer_onnx("onnx/lprnet.onnx")
    app.state.device = DEVICE
    app.state.conf = CONF_THRESHOLD
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(detect.router)
app.include_router(recognize.router)
app.include_router(pipeline.router)
app.include_router(video.router)
app.include_router(webcam.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://xtimate.github.io"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}
