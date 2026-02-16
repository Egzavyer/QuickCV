from models.Model import Model

imgs = ["006632"]

fm = Model("yolov8n.pt", 0.7, 0)
sm = Model("yolov8m.onnx", 0.7, 0)

for img in imgs:
    fm.predict(f"yolo/images/val/{img}.png")
    uncertain = fm.eval()
    fm.present()

    for i in range(len(uncertain)):
        sm.predict(uncertain[i][1])
        sm.present()
