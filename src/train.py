from ultralytics import YOLO  # type: ignore

model = YOLO("yolov8l.pt")

model.train(
    data="yolo/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device="cpu",
)
model.val()
