# Spotter

> Real-time license plate detection and recognition — built for Macondo.

**[Live Demo →](https://xtimate.github.io/torch-licenseplate/)** · **[Documentation →](https://github.com/Xtimate/torch-licenseplate/wiki)**

---

## What is it?

Spotter is an end-to-end license plate recognition system built from scratch. It combines a YOLOv8 detector with a custom-trained LPRNet recognizer, both exported to ONNX and served via a FastAPI backend. A SvelteKit frontend lets you test it in real time — upload an image, scan a video, or point your webcam at a plate.

---

## Quick start

**Backend**
```bash
git clone https://github.com/Xtimate/torch-licenseplate.git
cd torch-licenseplate
python -m venv venv && source venv/bin/activate  # or `venv\Scripts\activate` on Windows
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

Built for **Macondo** — a Hackclub project.
