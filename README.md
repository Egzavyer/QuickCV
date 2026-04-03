# Hybrid Object Detection for Autonomous Driving

A two-stage object detection pipeline for CPU-oriented deployment. The system first runs a fast YOLO detector on the full image, then selectively re-checks only low-confidence detections with a stronger second-stage model. The goal is to improve uncertain predictions without paying the cost of running the larger model on every object.

The project was trained and evaluated on a 4-class KITTI-style dataset with the following labels:

- Car
- Truck
- Pedestrian
- Cyclist

## Repository Structure

```text
.
├── main.py                    # Hybrid inference pipeline
├── models/
│   └── Model.py              # Wrapper around Ultralytics YOLO models
├── src/
│   ├── train.py              # Training script
│   ├── validate_models.py    # Validation script for trained models
│   ├── export.py             # ONNX/OpenVINO/OpenVINO INT8 export script
│   └── benchmark.py          # Warm-up + benchmark runner
├── benchmark_runs/           # Saved benchmark logs and summary table
├── requirements.txt
└── yolo/
    └── data.yaml             # Dataset config
```

## What the Pipeline Does

1. **Stage 1:** runs a fast detector on the full input image.
2. **Escalation:** selects detections below a confidence threshold.
3. **Stage 2:** crops a padded region around each uncertain detection and re-runs a stronger detector.
4. **Decision logic:** uses a conservative relabel rule so similar classes such as `Cyclist` and `Pedestrian` are not changed too easily.
5. **Outputs:** prints timing and confidence metrics and can save annotated images for visual inspection.

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Dataset Layout

The dataset itself is **not included** in this repository. Create a local `yolo/` directory with this structure:

```text
yolo/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Your `data.yaml` should point to those folders and define the 4 classes:

```yaml
train: images/train
val: images/val
nc: 4
names:
  - Car
  - Truck
  - Pedestrian
  - Cyclist
```

## Training

Train a YOLO model:

```bash
python src/train.py
```

Note: if your local dataset path differs from the one in `src/train.py`, update the `data=` argument first.

## Validation

Validate one or more trained models on the same dataset:

```bash
python src/validate_models.py \
  --models yolov8n.pt yolov8m.pt \
  --data yolo/data.yaml \
  --device cpu
```

This prints precision, recall, mAP50, mAP50-95, per-class mAP50-95, and validation speed.

## Exporting Models

Export trained `.pt` models to deployment formats:

```bash
python src/export.py \
  --models yolov8n.pt yolov8m.pt \
  --onnx \
  --openvino \
  --openvino-int8 \
  --data yolo/data.yaml \
  --imgsz 640
```

### Export notes

- Use **OpenVINO** for CPU-oriented deployment.
- Use **OpenVINO INT8** only after benchmarking on your target hardware.
- If you want stage 1 and stage 2 to use different input sizes, export them separately at those sizes.

Example:

```bash
python src/export.py --models yolov8n.pt --openvino-int8 --data yolo/data.yaml --imgsz 640
python src/export.py --models yolov8m.pt --openvino-int8 --data yolo/data.yaml --imgsz 320
```

## Running the Hybrid Pipeline

### Example with OpenVINO INT8 models

```bash
python main.py \
  --fast-model yolov8n_int8_640_openvino_model/ \
  --slow-model yolov8m_int8_320_openvino_model/ \
  --image-dir yolo/images/val \
  --images 006632 006638 006640 \
  --threshold 0.7 \
  --min-box-area 0 \
  --stage1-imgsz 640 \
  --stage2-imgsz 320 \
  --min-pad 32 \
  --pad-ratio 0.25 \
  --similar-min-conf 0.85 \
  --similar-min-delta 0.20 \
  --general-min-conf 0.75 \
  --general-min-delta 0.15 \
  --save-vis
```

### Important arguments

- `--fast-model`: model used for stage 1
- `--slow-model`: model used for stage 2
- `--threshold`: detections below this confidence are escalated
- `--stage1-imgsz`: image size for full-image inference
- `--stage2-imgsz`: image size for crop re-checks
- `--min-pad`: minimum crop padding in pixels
- `--pad-ratio`: additional crop padding as a fraction of box size
- `--save-vis`: saves annotated output images
- `--show`: opens GUI windows with annotated detections

## Benchmarking

Run a larger benchmark comparing regular OpenVINO and INT8 OpenVINO:

```bash
python src/benchmark.py \
  --main-script main.py \
  --image-dir yolo/images/val \
  --count 100 \
  --fast-model yolov8n_640_openvino_model/ \
  --slow-model yolov8m_320_openvino_model/ \
  --fast-model-int8 yolov8n_int8_640_openvino_model/ \
  --slow-model-int8 yolov8m_int8_320_openvino_model/ \
  --threshold 0.7 \
  --min-box-area 0 \
  --stage1-imgsz 640 \
  --stage2-imgsz 320 \
  --min-pad 32 \
  --pad-ratio 0.25 \
  --similar-min-conf 0.85 \
  --similar-min-delta 0.20 \
  --general-min-conf 0.75 \
  --general-min-delta 0.15 \
  --out-dir benchmark_runs
```

This produces:

- per-run log files
- warm-up logs
- `benchmark_runs/summary_table.md`

## Example Output

The pipeline prints:

- number of detections kept by stage 1
- number of escalations
- stage-1 and stage-2 inference time
- confidence gains after stage 2
- number of no-detection fallback failures
- end-to-end wall-clock time

Annotated images are saved to:

```text
runs/hybrid_vis/
```

## Current Best Deployment Configuration

On the tested CPU-oriented setup, the best-performing deployment format was:

- **Stage 1:** `yolov8n_int8_640_openvino_model/`
- **Stage 2:** `yolov8m_int8_320_openvino_model/`

## Limitations

- The dataset is not included in the repository.
- The pipeline was evaluated on CPU; results may differ on GPU or other hardware.
- The current label space contains only 4 classes.
- The hybrid system reports confidence-based improvements, but it is not a full autonomous driving perception stack.

## Reproducibility Notes

To reproduce the main results, include:

- the training scripts
- the dataset config (`yolo/data.yaml`)
- the final trained weights or a download link
- the benchmark logs in `benchmark_runs/`
- the final report PDF

If model files are too large for normal GitHub storage, use **Git LFS** or attach them to a GitHub Release instead of committing them directly.
