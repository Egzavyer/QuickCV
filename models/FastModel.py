from typing import Any
from ultralytics import YOLO  # type: ignore
import torch


class PredictionResult:
    def __init__(self, results) -> None:
        self.coords = results.boxes.xyxy
        self.class_ids = results.boxes.cls
        self.scores = results.boxes.conf


class FastModel:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.last_prediction = None
        self.MIN_CONFIDENCE_THRESHOLD = 0.6
        self.MIN_BOX_AREA = 700
        self.last_result = None
        self.skip_index = set()

        self.class_dict = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
            24: "backpack",
            25: "umbrella",
            26: "handbag",
            27: "tie",
            28: "suitcase",
            29: "frisbee",
            30: "skis",
            31: "snowboard",
            32: "sports ball",
            33: "kite",
            34: "baseball bat",
            35: "baseball glove",
            36: "skateboard",
            37: "surfboard",
            38: "tennis racket",
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            44: "spoon",
            45: "bowl",
            46: "banana",
            47: "apple",
            48: "sandwich",
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot dog",
            53: "pizza",
            54: "donut",
            55: "cake",
            56: "chair",
            57: "couch",
            58: "potted plant",
            59: "bed",
            60: "dining table",
            61: "toilet",
            62: "tv",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "cell phone",
            68: "microwave",
            69: "oven",
            70: "toaster",
            71: "sink",
            72: "refrigerator",
            73: "book",
            74: "clock",
            75: "vase",
            76: "scissors",
            77: "teddy bear",
            78: "hair drier",
            79: "toothbrush",
        }

    def predict(self, imgPath: str):
        self.last_prediction = self.model(imgPath)[0]
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
            keep_mask, device=self.last_prediction.boxes.data.device
        )
        filtered_data = self.last_prediction.boxes.data[keep_mask]
        self.last_prediction.update(boxes=filtered_data)

    def eval(self):
        if self.last_prediction is None:
            print("No prediction available. Call predict() first.")
            return

        if self.last_result is None:
            print("No result available. Call predict() first.")
            return

        for i in range(len(self.last_result.scores)):
            if i in self.skip_index:
                continue
            if self.last_result.scores[i] < self.MIN_CONFIDENCE_THRESHOLD:
                # TODO: send the coordinates of the bounding box to SlowModel
                print(
                    f"Unable to confidently classify object at: {self.last_result.coords[i]}"
                )

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
