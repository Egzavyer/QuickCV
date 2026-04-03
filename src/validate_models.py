from __future__ import annotations

import argparse
from time import perf_counter

from ultralytics import YOLO  # type: ignore


def run_validation(model_path: str, data_yaml: str, imgsz: int, split: str, device: str | None) -> None:
    print("\n" + "=" * 72)
    print(f"VALIDATING: {model_path}")
    print("=" * 72)

    model = YOLO(model_path)
    start = perf_counter()
    metrics = model.val(data=data_yaml, imgsz=imgsz, split=split, device=device)
    elapsed_ms = (perf_counter() - start) * 1000.0

    results_dict = getattr(metrics, "results_dict", {})
    for key, value in results_dict.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    names = getattr(metrics, "names", {})
    maps = getattr(metrics, "maps", None)
    if maps is not None and len(maps) == len(names):
        print("\nPer-class mAP50-95:")
        for class_index, class_map in enumerate(maps):
            class_name = names.get(class_index, str(class_index))
            print(f"  {class_name}: {float(class_map):.6f}")

    speed = getattr(metrics, "speed", {})
    if speed:
        print("\nValidation speed (ms/image):")
        for key, value in speed.items():
            print(f"  {key}: {float(value):.4f}")

    print(f"\nTotal validation wall-clock time: {elapsed_ms:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one or more YOLO models on the same dataset.")
    parser.add_argument("--models", nargs="+", required=True, help="Model paths to validate.")
    parser.add_argument("--data", required=True, help="Path to data.yaml.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--split", default="val", help="Dataset split to validate on.")
    parser.add_argument("--device", default=None, help="Ultralytics device string, for example cpu or 0.")
    args = parser.parse_args()

    for model_path in args.models:
        run_validation(
            model_path=model_path,
            data_yaml=args.data,
            imgsz=args.imgsz,
            split=args.split,
            device=args.device,
        )


if __name__ == "__main__":
    main()
