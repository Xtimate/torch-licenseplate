# Spotter

> Real-time license plate detection and recognition - built for Macondo.

**[Live Demo →](https://xtimate.github.io/torch-licenseplate/)**

---

## What is it?

Spotter is an end-to-end license palte recognition system built from scratch. It combines a YOLOv8 detector with a custom-trained LPRNet recognizer, both exported to ONNX and server via a FastAPI backend. A SvelteKit frontend lets you test it in real time -- upload an image, scan a video, or point your webcam at a plate.

No cloud APIs. No off-the-shelf OCR. Just a model trained on a mix of synthetic and real Dutch, German and French plates rendered with realistic augmentation.

---

## Features

- **Plate detection** via YOLOv8 (ONNX)
- **Text recognition** via custom LPRNet (ONNX)
- **Confidence scoring** - low-confidence results are rejected instead of returning garbage.
- **Fuzzy deduplication** - catches near-duplicates like '13BSRB' vs 'I3BSRB' using Levenshtein distance
- **Format validation** - recognizes and validates NL, DE and FR plate formats
- **Batch endpoint** - process multiple images in one request
- **Video scanning** - extract all unique plates across frames with the Levenshtein algorithm.
- **Live webcam** - real-time detection via WebSocket

---

## Stack

| Layer | Tech |
|---|---|
| Detection | YOLOv8 → ONNX Runtime |
| Recognition | Custom LPRNet → ONNX Runtime |
| Backend | FastAPI → Python |
| Frontend | SvelteKit + Tailwind |
| Training data | Synthetic plates via Pillow + Albumentations |
| Deployment | GitHub Pages + DigitalOcean

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---
| `POST` | `/pipeline` | Detect + recognize in one shot |
| `POST` | `/detect` | Detection only, returns bounding boxes |
| `POST` | `/recognize` | Recognition only on a cropped plate |
| `POST` | `/pipeline/batch` | Process multiple images at once |
| `POST` | `/video` | Scan a video file for unique plates |
| `WS` | `/webcam` | Live webcam stream |
| `GET` | `/health` | Health check |

---

## Running locally
```bash
**Backend**
git clone https://github.com/Xtimate/torch-licenseplate.git
cd torch-licenseplate
python -m venv venv && source venv/bin/activate # if you use a shell other than bash, look for the correct activate script in the bin folder, for example for fish: activate.fish
pip install -r requirements.txt
PYTHONPATH=src uvicorn api.main:app --reload
```

**Frontend**
```bash
cd spotter-ui
echo "VITE_API_BASE=http://localhost:8000" > .env.development
npm install && npm run dev
```

---

## Training your own model

Generate the dataset:
```bash
python src/generate_dataset.py --size 20000 --out data/plates --workers 8 #adjust the size to your likings, a minimum of 20000 images is recommended for the best results, if --workers 8 doesnt work, change it to 4 or 2 workers.
```

Train:
```bash
PYTHONPATH=src python src/train_recognizer.py
```

Export to ONNX:
```bash
python export_onnx.py
```

---

```
torch-licenseplate/
├── api/               # FastAPI app and routers
├── src/               # Dataset, generator, recognizer, detector
├── spotter-ui/        # SvelteKit frontend
├── onnx/              # Exported ONNX models
├── checkpoints/       # PyTorch checkpoints
└── detector-training/ # YOLOv8 training config
```

---

Built for **Macondo** - a HackClub project.
