import argparse
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_FIELDS = [
    "Invoice No",
    "Invoice Date",
    "Dealer Code",
    "Sale Order",
    "Vender Code",
    "Vehicle Code",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the KHB invoice YOLO field detector."
    )
    parser.add_argument(
        "--data",
        default="khb_dataset/data.yaml",
        help="Path to YOLO dataset data.yaml.",
    )
    parser.add_argument(
        "--base-model",
        default="yolov11n.pt",
        help="Base YOLOv11 model to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", default="khb_field_model_v2")
    parser.add_argument(
        "--project",
        default="runs/detect",
        help="Directory where training runs are saved.",
    )
    parser.add_argument(
        "--export-model",
        default="my_model/v2_best.pt",
        help="Copy the best YOLO checkpoint here after training. Use empty string to skip.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep existing YOLO labels.cache files. By default they are cleared.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Dataloader workers. 0 is safer on macOS/local notebooks.",
    )
    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Skip YOLO training after validating the dataset.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the dataset and exit without training.",
    )
    return parser.parse_args()


def clear_label_caches(dataset_dir):
    for cache_path in dataset_dir.glob("**/labels.cache"):
        cache_path.unlink()
        print(f"Removed stale cache: {cache_path}")


def parse_yaml_names(data_path):
    names = []
    inside_names = False
    for raw_line in data_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("names:"):
            inside_names = True
            inline = line.split(":", 1)[1].strip()
            if inline.startswith("[") and inline.endswith("]"):
                names = [
                    item.strip().strip("'\"")
                    for item in inline[1:-1].split(",")
                    if item.strip()
                ]
            continue
        if inside_names:
            if line.startswith("- "):
                names.append(line[2:].strip().strip("'\""))
            elif line and not raw_line.startswith((" ", "\t")):
                break
    return names


def validate_yolo_dataset(data_path):
    dataset_dir = data_path.parent
    names = parse_yaml_names(data_path)
    if names != DEFAULT_FIELDS:
        raise ValueError(
            "Dataset classes do not match the v2 invoice fields.\n"
            f"Expected: {DEFAULT_FIELDS}\n"
            f"Found:    {names}"
        )

    summary = {}
    for split in ("train", "valid", "test"):
        image_dir = dataset_dir / split / "images"
        label_dir = dataset_dir / split / "labels"
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing label directory: {label_dir}")

        images = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        labels = sorted(label_dir.glob("*.txt"))
        missing_labels = [
            image.name for image in images
            if not (label_dir / f"{image.stem}.txt").exists()
        ]
        if missing_labels:
            raise FileNotFoundError(
                f"{split} has images without labels: {', '.join(missing_labels[:10])}"
            )

        invalid_lines = []
        for label_path in labels:
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    invalid_lines.append(f"{label_path}:{line_number} expected 5 values")
                    continue
                class_id = int(float(parts[0]))
                values = [float(value) for value in parts[1:]]
                if class_id < 0 or class_id >= len(names):
                    invalid_lines.append(f"{label_path}:{line_number} invalid class {class_id}")
                if any(value < 0 or value > 1 for value in values):
                    invalid_lines.append(f"{label_path}:{line_number} bbox value outside 0..1")
        if invalid_lines:
            raise ValueError("Invalid YOLO labels:\n" + "\n".join(invalid_lines[:20]))

        summary[split] = (len(images), len(labels))

    print("Dataset OK:")
    for split, (image_count, label_count) in summary.items():
        print(f"  {split}: {image_count} images, {label_count} labels")


def copy_best_checkpoint(project_path, run_name, export_model):
    if not export_model:
        return
    best_path = project_path / run_name / "weights" / "best.pt"
    if not best_path.exists():
        print(f"Warning: best checkpoint was not found at {best_path}")
        return
    export_path = Path(export_model)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, export_path)
    print(f"Copied YOLO v2 checkpoint to: {export_path}")


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Create a YOLO dataset first, then run training."
        )

    validate_yolo_dataset(data_path)
    if args.validate_only:
        return

    if not args.keep_cache:
        clear_label_caches(data_path.parent)

    project_path = Path(args.project).resolve()

    if not args.skip_yolo:
        model = YOLO(args.base_model)
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=str(project_path),
            name=args.name,
            workers=args.workers,
            fliplr=0.0,
            flipud=0.0,
            mosaic=0.0,
            degrees=3.0,
            translate=0.05,
            scale=0.10,
            perspective=0.0005,
            close_mosaic=0,
        )
        copy_best_checkpoint(project_path, args.name, args.export_model)

if __name__ == "__main__":
    main()
