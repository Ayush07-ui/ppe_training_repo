from ultralytics import YOLO
import argparse


def main():
    parser = argparse.ArgumentParser(description="Export PPE compliance YOLO model")
    parser.add_argument("--weights", required=True,
                         help="Path to trained weights (.pt)")
    parser.add_argument("--format", default="onnx",
                         help="Export format: onnx, openvino, tflite, torchscript, etc.")
    parser.add_argument("--include", dest="format",
                         help="Alias for --format (pipeline compatibility)")
    parser.add_argument("--img", "--imgsz", dest="imgsz", type=int, default=640,
                         help="Inference image size")
    parser.add_argument("--data", default=None,
                         help="Path to dataset yaml (required by some export formats, e.g. int8/edge formats)")
    args = parser.parse_args()

    model = YOLO(args.weights)

    export_kwargs = dict(format=args.format, imgsz=args.imgsz)
    if args.data:
        export_kwargs["data"] = args.data

    exported_path = model.export(**export_kwargs)

    print("Export complete.")
    print(f"Exported model: {exported_path}")


if __name__ == "__main__":
    main()