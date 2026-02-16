from ultralytics import YOLO  # type: ignore

model = YOLO("yolov8m.pt")

model.export(format="onnx", dynamic=True)
