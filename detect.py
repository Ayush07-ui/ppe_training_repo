from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/train/ppe_compliance/weights/best.pt")
    parser.add_argument("--source", required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project="runs/detect",
        name="ppe_prediction",
    )

if __name__ == "__main__":
    main()
