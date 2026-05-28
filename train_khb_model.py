import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 model for KHB delivery-order field detection."
    )
    parser.add_argument(
        "--data",
        default="khb_dataset/data.yaml",
        help="Path to YOLO dataset data.yaml.",
    )
    parser.add_argument(
        "--base-model",
        default="yolov8n.pt",
        help="Base YOLOv8 model to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", default="khb_field_model")
    parser.add_argument(
        "--project",
        default="runs/detect",
        help="Directory where training runs are saved.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep existing YOLO labels.cache files. By default they are cleared.",
    )
    return parser.parse_args()


def clear_label_caches(dataset_dir):
    for cache_path in dataset_dir.glob("**/labels.cache"):
        cache_path.unlink()
        print(f"Removed stale cache: {cache_path}")


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Create a YOLO dataset first, then run training."
        )
    if not args.keep_cache:
        clear_label_caches(data_path.parent)

    project_path = Path(args.project).resolve()

    model = YOLO(args.base_model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project_path),
        name=args.name,
    )


if __name__ == "__main__":
    main()
