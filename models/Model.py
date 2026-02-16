from typing import Any
from ultralytics import YOLO  # type: ignore
import torch

# TODO: also check if the model is returning a result and if not handle that


class PredictionResult:
    def __init__(self, results) -> None:
        self.coords = results.boxes.xyxy
        self.class_ids = results.boxes.cls
        self.scores = results.boxes.conf


class Model:
    def __init__(self, model, minConf, minBox):
        self.model = YOLO(model)
        self.last_prediction = None
        self.MIN_CONFIDENCE_THRESHOLD = minConf
        self.MIN_BOX_AREA = minBox  # TODO: size of the bounding box is also affected by if the object is a car or truck or pedestrian or cyclist. This size should be dynamic based on the perceived class
        self.last_result = None
        self.skip_index = set()
        self.BOX_ENLARGEMENT = 75

        self.class_dict = {0: "Car", 1: "Truck", 2: "Pedestrian", 3: "Cyclist"}

    def predict(self, input):
        self.last_prediction = self.model(input)[0]
        self.last_result = PredictionResult(self.last_prediction)
        self.clean()
        return self.last_result

    def clean(self):
        if self.last_result is None:
            print("No result available. Call predict() first.")
            return

        if self.last_prediction is None:
            print("No prediction available. Call predict() first.")
            return

        keep_mask = []
        for i in range(len(self.last_result.scores)):
            if get_area(self.last_result.coords[i]) < self.MIN_BOX_AREA:
                self.skip_index.add(i)
                keep_mask.append(False)
            else:
                keep_mask.append(True)

        keep_mask = torch.tensor(
            keep_mask, dtype=torch.bool, device=self.last_prediction.boxes.data.device
        )
        filtered_data = self.last_prediction.boxes.data[keep_mask]
        self.last_prediction.update(boxes=filtered_data)

    def eval(self):
        crops = []
        if self.last_prediction is None:
            print("No prediction available. Call predict() first.")
            return crops

        if self.last_result is None:
            print("No result available. Call predict() first.")
            return crops

        for i in range(len(self.last_result.scores)):
            if i in self.skip_index:
                continue
            if self.last_result.scores[i] < self.MIN_CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = self.last_result.coords[i].cpu().numpy().astype(int)
                crop = self.last_prediction.orig_img[
                    y1 - self.BOX_ENLARGEMENT : y2 + self.BOX_ENLARGEMENT,
                    x1 - self.BOX_ENLARGEMENT : x2 + self.BOX_ENLARGEMENT,
                ]
                crops.append((i, crop))
                print(
                    f"Unable to confidently classify object at: {self.last_result.coords[i]}"
                )
        return crops

    def present(self):
        if self.last_prediction is None:
            print("No prediction available. Call predict() first.")
            return

        if self.last_result is None:
            print("No result available. Call predict() first.")
            return

        for box, cls_id, conf in zip(
            self.last_result.coords, self.last_result.class_ids, self.last_result.scores
        ):
            print(
                f"Class: {self.class_dict[int(cls_id)]}, Confidence: {conf:.2f}, Box: {box.tolist()}"
            )
        self.last_prediction.show()


def get_area(box):
    w = box[2] - box[0]
    h = box[3] - box[1]
    return w * h
