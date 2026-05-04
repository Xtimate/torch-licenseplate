import os

from dotenv import load_dotenv

load_dotenv()

DETECTOR_WEIGHTS = os.getenv("DETECTOR_WEIGHTS", "checkpoints/detector_best.pt")
RECOGNIZER_WEIGHTS = os.getenv("RECOGNIZER_WEIGHTS", "checkpoints/lprnet_best.pth")
DEVICE = os.getenv("DEVICE", "cpu")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.3"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "1.0"))
