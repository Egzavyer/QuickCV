from __future__ import annotations

import argparse

from ultralytics import YOLO  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a YOLO detector on the KITTI-format dataset."
    )
    parser.add_argument("--model", default="yolov8m.pt", help="Base model to fine-tune.")
    parser.add_argument("--data", default="yolo/data.yaml", help="Path to data.yaml.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        default="0",
        help="Ultralytics device string, e.g. '0' for GPU or 'cpu'.",
    )
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience.")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
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
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
