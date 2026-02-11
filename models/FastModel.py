from ultralytics import YOLO # type: ignore

class FastModel:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def predict(self, imgPath:str):
         return self.model(imgPath)
