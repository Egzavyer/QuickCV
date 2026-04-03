from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO  # type: ignore


def export_one(
    model_path: str,
    imgsz: int,
    export_onnx: bool,
    export_openvino: bool,
    export_openvino_int8: bool,
    data_yaml: str | None,
    dynamic: bool,
    half: bool,
) -> None:
    print("\n" + "=" * 72)
    print(f"EXPORTING: {model_path}")
    print("=" * 72)

    model = YOLO(model_path, task="detect")

    if export_onnx:
        print("\n[1/3] Exporting ONNX...")
        result = model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=dynamic,
            half=half,
        )
        print(f"Saved ONNX export: {result}")

    if export_openvino:
        print("\n[2/3] Exporting OpenVINO FP16/FP32...")
        result = model.export(
            format="openvino",
            imgsz=imgsz,
            half=half,
        )
        print(f"Saved OpenVINO export: {result}")

    if export_openvino_int8:
        if not data_yaml:
            raise ValueError("--data is required when using --openvino-int8")
        print("\n[3/3] Exporting OpenVINO INT8...")
        result = model.export(
            format="openvino",
            imgsz=imgsz,
            int8=True,
            data=data_yaml,
        )
        print(f"Saved OpenVINO INT8 export: {result}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YOLO detection models to ONNX and OpenVINO formats."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more .pt model paths to export.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Export image size.",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Export ONNX.",
    )
    parser.add_argument(
        "--openvino",
        action="store_true",
        help="Export standard OpenVINO model.",
    )
    parser.add_argument(
        "--openvino-int8",
        action="store_true",
        help="Export OpenVINO INT8 model using calibration data.",
    )
    parser.add_argument(
        "--data",
        help="Dataset YAML used for INT8 calibration.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic shape for ONNX export.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 where supported by the export backend.",
    )
    args = parser.parse_args()

    if not any([args.onnx, args.openvino, args.openvino_int8]):
        args.onnx = True
        args.openvino = True

    for model_path in args.models:
        if not Path(model_path).exists():
            print(f"Skipping missing model: {model_path}")
            continue

        export_one(
            model_path=model_path,
            imgsz=args.imgsz,
            export_onnx=args.onnx,
            export_openvino=args.openvino,
            export_openvino_int8=args.openvino_int8,
            data_yaml=args.data,
            dynamic=args.dynamic,
            half=args.half,
        )


if __name__ == "__main__":
    main()
