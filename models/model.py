"""YOLO/OpenVINO inference wrapper with confidence-gated escalation support."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch
from ultralytics import YOLO  # type: ignore

from .geometry import get_area, pad_box


class PredictionResult:
    def __init__(self, results) -> None:
        self.coords = (
            results.boxes.xyxy if results.boxes is not None else torch.empty((0, 4))
        )
        self.class_ids = (
            results.boxes.cls if results.boxes is not None else torch.empty((0,))
        )
        self.scores = (
            results.boxes.conf if results.boxes is not None else torch.empty((0,))
        )

    def __len__(self) -> int:
        return len(self.scores)


class Model:
    def __init__(
        self,
        model_path: str,
        min_confidence_threshold: float,
        min_box_area: float,
        box_enlargement: int = 32,
        crop_padding_ratio: float = 0.25,
    ) -> None:
        self.model = YOLO(model_path, task="detect")
        self.model_path = model_path
        self.last_prediction: Any = None
        self.last_result: PredictionResult | None = None

        self.MIN_CONFIDENCE_THRESHOLD = min_confidence_threshold
        self.MIN_BOX_AREA = min_box_area
        self.BOX_ENLARGEMENT = box_enlargement
        self.CROP_PADDING_RATIO = crop_padding_ratio
        self.last_inference_ms = 0.0

        model_names = getattr(self.model, "names", None)
        if isinstance(model_names, dict) and model_names:
            self.class_dict = model_names
        else:
            self.class_dict = {0: "Car", 1: "Truck", 2: "Pedestrian", 3: "Cyclist"}

    def predict(
        self,
        input_source: Any,
        imgsz: int | None = None,
        conf: float | None = None,
        device: str | None = None,
    ):
        predict_kwargs: dict[str, Any] = {"source": input_source, "verbose": False}
        if imgsz is not None:
            predict_kwargs["imgsz"] = imgsz
        if conf is not None:
            predict_kwargs["conf"] = conf
        if device is not None:
            predict_kwargs["device"] = device

        start = perf_counter()
        self.last_prediction = self.model.predict(**predict_kwargs)[0]
        self.last_inference_ms = (perf_counter() - start) * 1000.0

        self.last_result = PredictionResult(self.last_prediction)
        self.clean()
        return self.last_result

    def clean(self) -> None:
        if self.last_result is None or self.last_prediction is None:
            return

        if len(self.last_result) == 0:
            return

        boxes = self.last_prediction.boxes
        if boxes is None:
            return

        device = boxes.data.device
        keep_mask = torch.tensor(
            [get_area(box) >= self.MIN_BOX_AREA for box in self.last_result.coords],
            dtype=torch.bool,
            device=device,
        )
        self.last_prediction.update(boxes=boxes.data[keep_mask])
        self.last_result = PredictionResult(self.last_prediction)

    def eval(self) -> list[dict[str, Any]]:
        """Return crops that should be escalated to the second-stage model."""
        crops: list[dict[str, Any]] = []
        if self.last_prediction is None or self.last_result is None:
            return crops

        if len(self.last_result) == 0:
            return crops

        image = self.last_prediction.orig_img
        image_h, image_w = image.shape[:2]

        for i in range(len(self.last_result.scores)):
            score = float(self.last_result.scores[i])
            if score >= self.MIN_CONFIDENCE_THRESHOLD:
                continue

            box = self.last_result.coords[i].cpu().tolist()
            x1_pad, y1_pad, x2_pad, y2_pad = pad_box(
                box,
                image_width=image_w,
                image_height=image_h,
                min_pad=self.BOX_ENLARGEMENT,
                pad_ratio=self.CROP_PADDING_RATIO,
            )
            pad = max(
                self.BOX_ENLARGEMENT,
                int(round(max(box[2] - box[0], box[3] - box[1]) * self.CROP_PADDING_RATIO)),
            )

            if x2_pad <= x1_pad or y2_pad <= y1_pad:
                continue

            crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
            if crop.size == 0:
                continue

            class_id = int(self.last_result.class_ids[i])
            crops.append(
                {
                    "index": i,
                    "crop": crop,
                    "box": box,
                    "class_id": class_id,
                    "class_name": self.class_dict.get(class_id, str(class_id)),
                    "confidence": score,
                    "crop_xyxy": [x1_pad, y1_pad, x2_pad, y2_pad],
                    "pad_used": pad,
                }
            )

        return crops

    def best_detection(self) -> dict[str, Any] | None:
        result = self.last_result
        if result is None or len(result) == 0:
            return None

        best_index = max(
            range(len(result.scores)),
            key=lambda i: float(result.scores[i]),
        )
        best_class_id = int(result.class_ids[best_index])
        return {
            "index": best_index,
            "class_id": best_class_id,
            "class_name": self.class_dict.get(best_class_id, str(best_class_id)),
            "confidence": float(result.scores[best_index]),
            "box": result.coords[best_index].cpu().tolist(),
        }

    def detection_count(self) -> int:
        return len(self.last_result) if self.last_result is not None else 0

    def confidences(self) -> list[float]:
        if self.last_result is None:
            return []
        return [float(score) for score in self.last_result.scores]

    def annotated_image(self):
        if self.last_prediction is None:
            return None
        return self.last_prediction.plot()

    def present(self, show: bool = False) -> None:
        if self.last_prediction is None or self.last_result is None:
            print("No prediction available. Call predict() first.")
            return

        if len(self.last_result) == 0:
            print("No detections.")
            return

        for box, cls_id, conf in zip(
            self.last_result.coords,
            self.last_result.class_ids,
            self.last_result.scores,
            strict=False,
        ):
            class_id = int(cls_id)
            print(
                f"Class: {self.class_dict.get(class_id, class_id)}, "
                f"Confidence: {float(conf):.2f}, "
                f"Box: {box.tolist()}"
            )

        if show:
            self.last_prediction.show()
