# PPE Compliance YOLO Training Repository

This repository is prepared for a PPE compliance object-detection pipeline.

## Classes

| ID | Class |
|---:|---|
| 0 | person |
| 1 | helmet |
| 2 | no_helmet |
| 3 | vest |
| 4 | no_vest |

## Important: annotation is still required

The uploaded ZIP contained images only. It did not contain YOLO label `.txt` files.

Before training, annotate the images in CVAT and export YOLO-format labels.

Expected structure:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   ├── test/
│   └── raw/
└── labels/
    ├── train/
    ├── val/
    ├── test/
    └── raw/
```

Each image should have a matching label file:

```text
00001.jpg
00001.txt
```

A YOLO label line is:

```text
class_id x_center y_center width height
```

All coordinates are normalized from 0 to 1.

## Prepare the dataset

After placing CVAT-exported labels in `dataset/labels/raw/`:

```bash
python scripts/split_dataset.py
```

The script creates an 80/10/10 train/validation/test split.

## Install

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Then:

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

Default configuration:

- Model: `yolov8n.pt`
- Epochs: `100`
- Image size: `640`
- Batch size: `16`

For example:

```bash
python train.py --epochs 100 --imgsz 640 --batch 16
```

The best checkpoint will normally be:

```text
runs/train/ppe_compliance/weights/best.pt
```

## Validate

```bash
python validate.py
```

## Run detection

```bash
python detect.py --source path/to/image.jpg
```

Example:

```bash
python detect.py --source test.jpg --conf 0.5
```

## Export

ONNX:

```bash
python export.py --format onnx
```

OpenVINO:

```bash
python export.py --format openvino
```

TFLite:

```bash
python export.py --format tflite
```

## TrainX repository

TrainX expects the training code/configuration to live in the Git repository. The repository URL should point to this project's Git repository, for example:

```text
https://github.com/YOUR_USERNAME/ppe-compliance-training.git
```

After pushing this repository to GitHub, paste that repository's clone URL into TrainX.

Do not put the local ZIP path into TrainX.

## Recommended workflow

```text
Raw images
    ↓
CVAT annotation
    ↓
YOLO labels
    ↓
split_dataset.py
    ↓
Train
    ↓
Validate
    ↓
best.pt
    ↓
Export
    ↓
TrainX model pipeline
```
