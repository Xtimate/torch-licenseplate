from ultralytics import YOLO

model = YOLO("checkpoints/detector_best.pt")
model.export(format="onnx", dynamic=True, simplify=True)
