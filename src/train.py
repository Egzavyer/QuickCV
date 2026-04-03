from ultralytics import YOLO  # type: ignore

model = YOLO("yolov8m.pt")

model.train(
    data="yolo/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    # Optimization
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    # Regularization
    dropout=0.0,
    # Early stopping
    patience=20,
)
