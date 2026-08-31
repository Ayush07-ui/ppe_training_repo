from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/train/ppe_compliance/weights/best.pt")
    parser.add_argument("--format", default="onnx",
                        choices=["onnx", "openvino", "tflite", "torchscript"])
    args = parser.parse_args()

    model = YOLO(args.weights)
    path = model.export(format=args.format)
    print(f"Exported model: {path}")

if __name__ == "__main__":
    main()
