from models.FastModel import FastModel
from pathlib import Path
from labelformat.formats import KittiObjectDetectionInput, YOLOv8ObjectDetectionOutput

# Load KITTI labels
label_input = KittiObjectDetectionInput(
    input_folder=Path("kitti/training/labels"),
    category_names="car,truck,pedestrian,cyclist",
    images_rel_path="../images"
)

# Convert to YOLOv8 and save
YOLOv8ObjectDetectionOutput(
    output_file=Path("yolo/yolo.yaml"),
    output_split="train"
).save(label_input=label_input)

#fm = FastModel()

#results = fm.predict("data/car.jpeg")
#for result in results:
#    print(result)
#    result.show()