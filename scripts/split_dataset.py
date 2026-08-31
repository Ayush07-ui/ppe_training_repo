"""
Split annotated YOLO data into train/val/test.

Expected input:
    dataset/images/raw/*.jpg
    dataset/labels/raw/*.txt

Run:
    python scripts/split_dataset.py

Default split:
    80% train
    10% val
    10% test

Images without matching labels are skipped and reported.
"""

from pathlib import Path
import random
import shutil

ROOT = Path("dataset")
IMAGE_RAW = ROOT / "images" / "raw"
LABEL_RAW = ROOT / "labels" / "raw"
SEED = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    random.seed(SEED)

    if not IMAGE_RAW.exists():
        raise SystemExit(f"Missing {IMAGE_RAW}")

    pairs = []
    missing = []

    for image in sorted(IMAGE_RAW.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTS:
            continue
        label = LABEL_RAW / f"{image.stem}.txt"
        if label.exists():
            pairs.append((image, label))
        else:
            missing.append(image.name)

    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train+n_val],
        "test": pairs[n_train+n_val:],
    }

    for split, items in splits.items():
        for image, label in items:
            shutil.copy2(image, ROOT / "images" / split / image.name)
            shutil.copy2(label, ROOT / "labels" / split / label.name)

    print(f"Annotated pairs: {n}")
    print(f"Train: {len(splits['train'])}")
    print(f"Val:   {len(splits['val'])}")
    print(f"Test:  {len(splits['test'])}")
    print(f"Images without labels: {len(missing)}")

if __name__ == "__main__":
    main()
