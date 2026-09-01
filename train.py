from ultralytics import YOLO
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train PPE compliance YOLO detector")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="Path or name of the base checkpoint/model to train from")
    parser.add_argument("--weights", dest="model",
                         help="Alias for --model (kept for pipeline/legacy compatibility)")
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="ppe_compliance")
    parser.add_argument("--seed", type=int, default=0,
                         help="Random seed for reproducibility")
    parser.add_argument("--exist-ok", action="store_true",
                         help="Allow overwriting an existing project/name directory")
    args = parser.parse_args()

    model = YOLO(args.model)

    kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        pretrained=True,
        verbose=True,
        seed=args.seed,
        exist_ok=args.exist_ok,
    )
    if args.device:
        kwargs["device"] = args.device

    results = model.train(**kwargs)

    print("Training complete.")
    print(f"Results: {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()