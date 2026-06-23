# KITTI 2D Object Detection — YOLO Format (4 classes)

A YOLO-formatted version of the [KITTI 2D Object Detection benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php),
reduced to four driving-relevant classes and split for training/validation. Used by the
[Hybrid Object Detection for Autonomous Driving](https://github.com/Egzavyer/QuickCV) project.

## Contents

```text
yolo/
├── data.yaml
├── images/
│   ├── train/   # 5,984 images
│   └── val/     # 1,497 images
└── labels/
    ├── train/
    └── val/
```

- **7,481 images total** (80 / 20 train / val split).
- Labels in YOLO format: one `.txt` per image, each line `class_id x_center y_center width height` with coordinates normalized to `[0, 1]`.

## Classes

| ID | Class      | Instances |
| -- | ---------- | --------- |
| 0  | Car        | 28,742    |
| 1  | Truck      | 1,094     |
| 2  | Pedestrian | 4,487     |
| 3  | Cyclist    | 1,627     |
|    | **Total**  | **35,950**|

The dataset is class-imbalanced (vehicles dominate, trucks are rare), which is typical of
on-road capture and is reflected in lower per-class accuracy for the minority classes.

## License & Attribution

Derived from the KITTI Vision Benchmark Suite, released under
**CC BY-NC-SA 3.0** (non-commercial). Please cite:

> A. Geiger, P. Lenz, and R. Urtasun. *Are we ready for Autonomous Driving? The KITTI
> Vision Benchmark Suite.* CVPR, 2012.

## Usage

```bash
python src/download_dataset.py
```

This places the data in the layout above, matching `yolo/data.yaml`. See the
[project README](https://github.com/Egzavyer/QuickCV#readme) for training, validation,
export, and inference instructions.
