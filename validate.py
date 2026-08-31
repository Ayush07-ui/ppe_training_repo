from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/train/ppe_compliance/weights/best.pt")
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz)
    print(metrics)

if __name__ == "__main__":
    main()
