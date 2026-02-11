from ultralytics import YOLO  # type: ignore

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/yolo/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device="cpu",
)
model.val()
