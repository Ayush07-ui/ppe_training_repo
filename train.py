from ultralytics import YOLO
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train PPE compliance YOLO detector")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="ppe_compliance")
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
    )
    if args.device:
        kwargs["device"] = args.device

    results = model.train(**kwargs)
    print("Training complete.")
    print(f"Results: {Path(args.project) / args.name}")

if __name__ == "__main__":
    main()
