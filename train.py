"""
TrainX YOLO training entry point.

Supports:
- YOLOv5: delegates to the shared YOLOv5 checkout.
- YOLOv8 / YOLO11 / YOLO26: trains through ultralytics.

The TrainX pipeline requires these command-line flags:
    --img --batch --epochs --data --weights --project --name
    --seed --exist-ok

The pipeline also requires:
    <project>/<name>/results.csv
    <project>/<name>/weights/best.pt
"""

import argparse
import csv
import os
import shutil
import sys
import zipfile


# Shared YOLOv5 checkout used by the TrainX host.
YOLOV5_ROOT_DEFAULT = "/home/ivis/projects/Human-Detection/code"

# Model families handled directly by the ultralytics package.
ULTRALYTICS_PREFIXES = ("yolov8", "yolo11", "yolo26")

# TrainX expects YOLOv5-style metric names.
RESULTS_CSV_COLUMN_MAP = {
    "metrics/precision(B)": "metrics/precision",
    "metrics/recall(B)": "metrics/recall",
    "metrics/mAP50(B)": "metrics/mAP_0.5",
    "metrics/mAP50-95(B)": "metrics/mAP_0.5:0.95",
}


def parse_args():
    """Parse the exact CLI contract used by the TrainX orchestrator."""
    parser = argparse.ArgumentParser(
        description="TrainX multi-architecture PPE YOLO training entry point"
    )

    # Keep these names compatible with run_training.sh.
    parser.add_argument(
        "--img", "--imgsz",
        type=int,
        default=320,
        help="Training image size"
    )
    parser.add_argument(
        "--batch", "--batch-size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Generated dataset.yaml"
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Starting model checkpoint or model name"
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Parent directory for training output"
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Training run directory name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow an existing run directory"
    )

    return parser.parse_args()


def fail(message):
    """Print a clear pipeline-friendly error and return failure."""
    print(f"[train.py] ERROR: {message}", file=sys.stderr)
    return 1


def sniff_checkpoint_family(weights):
    """
    Identify an existing PyTorch checkpoint without loading the model.

    Ultralytics checkpoints contain 'ultralytics' module references.
    YOLOv5 checkpoints contain 'models.yolo' module references.
    """
    try:
        with zipfile.ZipFile(weights) as zf:
            pickle_file = next(
                (name for name in zf.namelist() if name.endswith("data.pkl")),
                None
            )

            if not pickle_file:
                return None

            raw = zf.read(pickle_file)

    except Exception:
        return None

    if b"ultralytics" in raw:
        return "ultralytics"

    if b"models.yolo" in raw:
        return "yolov5"

    return None


def weights_family(weights):
    """
    Determine the model family.

    First use the filename for normal pretrained models.
    If the filename is something generic such as best.pt, inspect the
    checkpoint contents.
    """
    base = os.path.basename(str(weights)).lower()

    if base.startswith("yolov5"):
        return "yolov5", f"basename {base!r}"

    if base.startswith(ULTRALYTICS_PREFIXES):
        return "ultralytics", f"basename {base!r}"

    if os.path.isfile(weights):
        family = sniff_checkpoint_family(weights)

        if family:
            return family, f"checkpoint contents of {base!r}"

    return None, f"basename {base!r}"


def train_yolov5_delegate(argv):
    """
    Delegate YOLOv5 training to the shared YOLOv5 train.py.

    The original command-line arguments are passed unchanged so the
    orchestrator's YOLOv5 CLI contract remains intact.
    """
    root = os.environ.get("YOLOV5_ROOT") or YOLOV5_ROOT_DEFAULT
    target = os.path.join(root, "train.py")

    if not os.path.isfile(target):
        if os.environ.get("YOLOV5_ROOT"):
            via = f"YOLOV5_ROOT={os.environ['YOLOV5_ROOT']}"
        else:
            via = (
                f"YOLOV5_ROOT is unset; "
                f"tried default {YOLOV5_ROOT_DEFAULT}"
            )

        return fail(
            "YOLOv5 training was selected, but the shared YOLOv5 "
            f"train.py does not exist at {target} ({via}). "
            "Set YOLOV5_ROOT to a valid YOLOv5 checkout."
        )

    # Replace this process with the official YOLOv5 training process.
    os.execv(sys.executable, [sys.executable, target] + argv)


def rewrite_results_csv(run_dir):
    """
    Convert ultralytics results.csv metric names into the names expected
    by the TrainX acceptance gate.
    """
    path = os.path.join(run_dir, "results.csv")

    if not os.path.isfile(path):
        return fail(
            f"training finished but {path} was not written. "
            "The TrainX pipeline requires results.csv."
        )

    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    if not rows:
        return fail(
            f"{path} is empty. The pipeline requires per-epoch metrics."
        )

    header = [
        RESULTS_CSV_COLUMN_MAP.get(column.strip(), column.strip())
        for column in rows[0]
    ]

    body = [
        [column.strip() for column in row]
        for row in rows[1:]
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(body)

    print(f"[train.py] Rewrote metrics in {path}")
    return 0


def ensure_best_pt(run_dir):
    """
    Make sure weights/best.pt exists.

    Ultralytics normally creates best.pt. If only last.pt exists,
    use last.pt as the final checkpoint.
    """
    weights_dir = os.path.join(run_dir, "weights")
    best = os.path.join(weights_dir, "best.pt")
    last = os.path.join(weights_dir, "last.pt")

    if os.path.isfile(best):
        return 0

    if os.path.isfile(last):
        shutil.copy2(last, best)
        print(f"[train.py] Copied last.pt -> {best}")
        return 0

    return fail(
        "training finished but neither best.pt nor last.pt exists "
        f"under {weights_dir}. The pipeline requires weights/best.pt."
    )


def train_ultralytics(args):
    """Train YOLOv8/YOLO11/YOLO26 using the ultralytics package."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        return fail(
            "The ultralytics package is not installed or cannot be imported: "
            f"{error}. Install ultralytics in the training environment."
        )

    # If the path exists, use it directly.
    # Otherwise pass only the filename so ultralytics can download a
    # pretrained model such as yolov8n.pt.
    weights = (
        args.weights
        if os.path.isfile(args.weights)
        else os.path.basename(args.weights)
    )

    print(f"[train.py] Loading model: {weights}")
    model = YOLO(weights)

    # Train using the arguments supplied by the TrainX orchestrator.
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        project=args.project,
        name=args.name,
        seed=args.seed,
        exist_ok=args.exist_ok,
    )

    # Ultralytics normally writes to project/name.
    run_dir = os.path.join(args.project, args.name)

    # Prefer the actual directory reported by the ultralytics trainer.
    save_dir = getattr(
        getattr(model, "trainer", None),
        "save_dir",
        None
    )

    if save_dir:
        run_dir = str(save_dir)

    print(f"[train.py] Training run directory: {run_dir}")

    # TrainX acceptance gate expects YOLOv5 metric column names.
    result = rewrite_results_csv(run_dir)

    if result:
        return result

    # TrainX export/warm-start stage requires best.pt.
    return ensure_best_pt(run_dir)


def main():
    """Main entry point used by the TrainX training pipeline."""
    args = parse_args()

    family, reason = weights_family(args.weights)

    print(
        f"[train.py] weights={args.weights} -> "
        f"family={family or 'UNKNOWN'} ({reason})"
    )

    if family == "yolov5":
        return train_yolov5_delegate(sys.argv[1:])

    if family == "ultralytics":
        return train_ultralytics(args)

    return fail(
        f"Cannot determine model family from --weights "
        f"{args.weights!r}. Supported model families are YOLOv5, "
        "YOLOv8, YOLO11, and YOLO26. Existing best.pt checkpoints "
        "are identified from their checkpoint contents."
    )


if __name__ == "__main__":
    sys.exit(main())
