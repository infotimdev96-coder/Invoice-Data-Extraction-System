import argparse
import contextlib
import io
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import cv2
from ultralytics import YOLO


FIELDS = [
    "Invoice No",
    "Invoice Date",
    "Dealer Code",
    "Sale Order",
    "Vender Code",
    "Vehicle Code",
]

FIELD_SET = set(FIELDS)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_MODEL_PATH = "runs/detect/khb_field_model_6fields/weights/best.pt"
DEFAULT_CROP_DIR = "crops"


def load_yolo_model(model_path):
    """Load a YOLO model with suppressed output."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return YOLO(str(model_path))


@lru_cache(maxsize=4)
def get_cached_model(model_path):
    """Load and cache YOLO models."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return load_yolo_model(path)


def rotate_image(image, angle):
    """Rotate an image by the specified angle."""
    if angle == 0:
        return image
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def save_cropped_image(crop, output_dir, filename, field_name, index=0):
    """Save a YOLO field crop."""
    if crop is None or crop.size == 0:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_field_name = field_name.replace(" ", "_")
    base_name = Path(filename).stem
    crop_path = output_dir / f"{base_name}_{safe_field_name}_{index}.png"

    if not cv2.imwrite(str(crop_path), crop):
        raise OSError(f"Could not save cropped image: {crop_path}")
    return crop_path


def detect_fields(image, model, confidence):
    """Detect invoice fields using the YOLO model."""
    results = model.predict(image, conf=confidence, verbose=False)
    names = model.names

    best_by_field = {}
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        field_name = names[class_id]
        if field_name not in FIELD_SET:
            continue

        xmin, ymin, xmax, ymax = [int(value) for value in box.xyxy[0].tolist()]
        score = float(box.conf[0])
        crop = image[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            continue

        detection = {
            "field_name": field_name,
            "confidence": score,
            "bbox": [xmin, ymin, xmax, ymax],
            "crop": crop,
        }
        if field_name not in best_by_field or score > best_by_field[field_name]["confidence"]:
            best_by_field[field_name] = detection

    return best_by_field


def extract_invoice_fields(
    image,
    model,
    conf=0.25,
    rotate_angle=0.0,
    crop_dir=None,
    filename=None,
):
    """Detect invoice fields and optionally save their YOLO crops."""
    image = rotate_image(image, rotate_angle)
    detections = detect_fields(image, model, conf)

    detected_fields = []
    saved_crops = []
    for index, field_name in enumerate(FIELDS):
        detection = detections.get(field_name)
        if not detection:
            continue

        detected_fields.append(
            {
                "field_name": field_name,
                "confidence": detection["confidence"],
                "bbox": detection["bbox"],
            }
        )
        if crop_dir and filename:
            crop_path = save_cropped_image(
                detection["crop"],
                crop_dir,
                filename,
                field_name,
                index,
            )
            if crop_path:
                saved_crops.append(
                    {
                        "field_name": field_name,
                        "path": str(crop_path),
                    }
                )

    return {
        "detected_fields": detected_fields,
        "missing_fields": [field_name for field_name in FIELDS if field_name not in detections],
        "saved_crops": saved_crops,
    }


def extract_from_file(
    image_path,
    model_path=DEFAULT_MODEL_PATH,
    conf=0.25,
    rotate_angle=0.0,
    crop_dir=DEFAULT_CROP_DIR,
):
    """Detect invoice fields in an image file."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    model = get_cached_model(model_path)
    result = extract_invoice_fields(
        image,
        model,
        conf,
        rotate_angle,
        crop_dir,
        Path(image_path).name,
    )
    return {
        "filename": Path(image_path).name,
        **result,
    }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Detect invoice fields using a YOLO model")
    parser.add_argument("--image_path", "--image", default="images", help="Path to image or folder")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--rotate", type=float, default=0.0, help="Rotate image angle")
    parser.add_argument(
        "--crop-dir",
        default=DEFAULT_CROP_DIR,
        help="Directory to save YOLO field crops. Use an empty value to disable saving.",
    )
    return parser.parse_args()


def find_image_paths(image_path):
    """Find all supported image files in a path."""
    path = Path(image_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        image_paths = [
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not image_paths:
            raise FileNotFoundError(f"No supported images found in: {path}")
        return sorted(image_paths)
    raise FileNotFoundError(f"Image path does not exist: {path}")


def main():
    """Run YOLO field detection for one image or a folder of images."""
    args = parse_args()
    results = []
    for image_path in find_image_paths(args.image_path):
        try:
            result = extract_from_file(
                image_path,
                args.model,
                args.conf,
                args.rotate,
                args.crop_dir or None,
            )
            results.append(result)
            print(f"Processed: {result['filename']}")
            print(f"  Detected: {[field['field_name'] for field in result['detected_fields']]}")
            print(f"  Missing: {result['missing_fields']}")
            if result["saved_crops"]:
                print(f"  Saved crops ({len(result['saved_crops'])}):")
                for crop in result["saved_crops"]:
                    print(f"    - {crop['field_name']}: {crop['path']}")
            print()
        except Exception as exc:
            print(f"Error processing {image_path}: {exc}")

    print(json.dumps({"count": len(results), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
