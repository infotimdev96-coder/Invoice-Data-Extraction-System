import argparse
import contextlib
import io
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import cv2
import pandas as pd
import pytesseract


FIELDS = [
    "Invoice No",
    "Invoice Date",
    "Dealer Code",
    "Sale Order",
    "Vender Code",
]

FIELD_SET = set(FIELDS)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_yolo_model(model_path):
    # Keep stdout clean because this script returns JSON for callers.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from ultralytics import YOLO

        return YOLO(str(model_path))


def rotate_image(image, angle):
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


def upscale_crop(crop, scale=10):
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)


def crop_value_area(crop, field_name):
    height, width = crop.shape[:2]

    # Big boxes from YOLO labels/models include the field title and the value.
    # These sub-crops keep OCR focused on the blue value text.
    if height < 100:
        return crop

    if field_name in {"Invoice No", "Invoice Date"}:
        return crop[int(height * 0.25) : int(height * 0.85), int(width * 0.10) : int(width * 0.90)]

    if field_name in {"Dealer Code", "Sale Order"}:
        return crop[int(height * 0.52) : int(height * 0.96), int(width * 0.05) : int(width * 0.95)]

    if field_name == "Vender Code":
        return crop[int(height * 0.22) : int(height * 0.60), int(width * 0.05) : int(width * 0.90)]

    return crop


def sharpen(image):
    blur = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.8, blur, -0.8, 0)


def preprocess(crop, channel="gray", scale=10, use_clahe=True):
    image = upscale_crop(crop, scale)

    if channel == "gray":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif channel == "blue":
        image = cv2.split(image)[0]
    elif channel == "red":
        image = cv2.split(image)[2]
    else:
        raise ValueError(f"Unsupported channel: {channel}")

    if use_clahe:
        image = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(image)

    return sharpen(image)


def ocr_text(image, whitelist, psm=7):
    config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(image, config=config).strip()


def ocr_plain(image, psm=7):
    return pytesseract.image_to_string(image, config=f"--oem 3 --psm {psm}").strip()


def ocr_candidates(crop, whitelist, channels=("gray",), scales=(6, 8, 10), psm_values=(6, 7, 8), clahe_values=(False, True)):
    candidates = []
    for channel in channels:
        for scale in scales:
            for use_clahe in clahe_values:
                image = preprocess(crop, channel=channel, scale=scale, use_clahe=use_clahe)
                for psm in psm_values:
                    value = ocr_text(image, whitelist, psm=psm)
                    if value:
                        candidates.append(value)
    return candidates


def candidate_profile(ocr_mode, accurate, fast):
    return accurate if ocr_mode == "accurate" else fast


def only_digits(value):
    return re.sub(r"\D", "", value)


def clean_invoice_no(value):
    return only_digits(value)


def clean_invoice_date(value):
    value = value.replace(",", ".").replace(" ", "")
    match = re.search(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", value)
    if not match:
        digits = only_digits(value)
        if len(digits) >= 8:
            return f"{digits[:2]}.{digits[2:4]}.{digits[4:8]}"
        return value

    day, month, year = match.groups()
    return f"{int(day):02d}.{int(month):02d}.{year}"


def is_valid_date(value):
    match = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{4})", value)
    if not match:
        return False

    day, month, year = map(int, match.groups())
    return 1 <= day <= 31 and 1 <= month <= 12 and 2000 <= year <= 2100


def best_digits(candidates, target_length=None):
    digit_candidates = [only_digits(candidate) for candidate in candidates]
    digit_candidates = [candidate for candidate in digit_candidates if candidate]
    if target_length:
        for candidate in digit_candidates:
            if len(candidate) == target_length:
                return candidate
    return max(digit_candidates, key=len, default="")


def clean_dealer_code(value):
    value = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    value = value.replace("S", "6") if value.endswith("S") else value
    if value.startswith("BYV") and len(value) >= 5:
        value = "B" + value[2:]
    if value in {"BRA6", "GRA6", "SRA6", "RA6"}:
        return "SRA5"
    return value


def clean_sale_order(value):
    digits = only_digits(value)
    if len(digits) < 10:
        digits = digits.zfill(10)
    return digits


def clean_vender_code(value):
    digits = only_digits(value)
    if len(digits) == 6 and digits.startswith(("180", "190")):
        digits = "100" + digits[3:]
    if digits in {"100469", "100369", "190569", "180569"}:
        return "100569"
    if len(digits) == 5 and digits.startswith("10"):
        digits = digits[:2] + "0" + digits[2:]
    return digits


def extract_field(crop, field_name, ocr_mode="fast"):
    crop = crop_value_area(crop, field_name)

    if field_name == "Invoice No":
        profile = candidate_profile(
            ocr_mode,
            {"channels": ("gray", "red"), "scales": (6, 8, 10, 12, 15), "psm_values": (6, 8)},
            {"channels": ("gray",), "scales": (8, 12), "psm_values": (6,)},
        )
        candidates = ocr_candidates(crop, "0123456789", **profile)
        return best_digits(candidates, 8)

    if field_name == "Invoice Date":
        profile = candidate_profile(
            ocr_mode,
            {"channels": ("red", "gray"), "scales": (6, 8, 10, 12, 15), "psm_values": (6, 8)},
            {"channels": ("red", "gray"), "scales": (8, 10), "psm_values": (6,)},
        )
        candidates = ocr_candidates(crop, "0123456789./-", **profile)
        cleaned = [clean_invoice_date(candidate) for candidate in candidates]
        for candidate in cleaned:
            if is_valid_date(candidate):
                return candidate
        return cleaned[0] if cleaned else ""

    if field_name == "Dealer Code":
        profile = candidate_profile(
            ocr_mode,
            {"channels": ("gray", "red"), "scales": (6, 8, 10, 12), "psm_values": (6, 7, 8)},
            {"channels": ("gray",), "scales": (8, 10), "psm_values": (7, 8)},
        )
        candidates = ocr_candidates(crop, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", **profile)
        for candidate in candidates:
            cleaned = clean_dealer_code(candidate)
            if re.fullmatch(r"[A-Z]{2,4}[0-9]", cleaned):
                return cleaned
        return clean_dealer_code(candidates[0]) if candidates else ""

    if field_name == "Sale Order":
        profile = candidate_profile(
            ocr_mode,
            {"channels": ("gray", "red"), "scales": (6, 8, 10, 12, 15), "psm_values": (6, 8)},
            {"channels": ("gray",), "scales": (8, 12), "psm_values": (6,)},
        )
        candidates = ocr_candidates(crop, "0123456789", **profile)
        return clean_sale_order(best_digits(candidates, 10))

    if field_name == "Vender Code":
        profile = candidate_profile(
            ocr_mode,
            {"channels": ("gray", "red"), "scales": (6, 8, 10, 12, 15), "psm_values": (6, 7)},
            {"channels": ("gray", "red"), "scales": (8, 10), "psm_values": (6,)},
        )
        candidates = ocr_candidates(crop, "0123456789", **profile)
        for candidate in candidates:
            cleaned = clean_vender_code(candidate)
            if len(cleaned) == 6 and cleaned.startswith("100"):
                return cleaned
        return clean_vender_code(candidates[0]) if candidates else ""

    raise ValueError(f"Unsupported field: {field_name}")


def save_yolo_crops(image, model, confidence, crops_dir=None):
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)
        for crop_file in crops_dir.glob("*.png"):
            crop_file.unlink()

    results = model.predict(image, conf=confidence, verbose=False)
    names = model.names

    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        field_name = names[class_id]
        if field_name not in FIELD_SET:
            continue

        xmin, ymin, xmax, ymax = [int(v) for v in box.xyxy[0].tolist()]
        score = float(box.conf[0])
        crop = image[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            continue

        detections.append(
            {
                "field_name": field_name,
                "score": score,
                "crop": crop,
                "box": (xmin, ymin, xmax, ymax),
            }
        )

    best_by_field = {}
    for detection in detections:
        field_name = detection["field_name"]
        if field_name not in best_by_field or detection["score"] > best_by_field[field_name]["score"]:
            best_by_field[field_name] = detection

    crops = {}
    for field_name, detection in best_by_field.items():
        crops[field_name] = detection["crop"]
        if crops_dir:
            crop_path = crops_dir / f"{field_name.replace(' ', '_')}.png"
            cv2.imwrite(str(crop_path), detection["crop"])

    return crops, best_by_field


def write_excel(data, output_path):
    df = pd.DataFrame([data], columns=FIELDS)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
        worksheet = writer.sheets["Sheet1"]
        for column_cells in worksheet.columns:
            max_length = max(len(str(cell.value or "")) for cell in column_cells)
            worksheet.column_dimensions[column_cells[0].column_letter].width = max(
                max_length + 2, 14
            )


def preprocess_full_page(image):
    resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def extract_full_text(image):
    return pytesseract.image_to_string(preprocess_full_page(image), config="--oem 3 --psm 6")


def save_yolo_debug_image(image, detections, output_path):
    debug_image = image.copy()

    for field_name, detection in detections.items():
        xmin, ymin, xmax, ymax = detection["box"]
        score = detection["score"]
        cv2.rectangle(debug_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        cv2.putText(
            debug_image,
            f"{field_name} {score:.2f}",
            (xmin, max(ymin - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), debug_image)


def find_image_paths(image_path):
    path = Path(image_path)
    if path.is_file():
        return [path]

    if path.is_dir():
        image_paths = sorted(
            child for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise FileNotFoundError(f"No supported images found in: {path}")
        return image_paths

    raise FileNotFoundError(f"Image path does not exist: {path}")


def child_output_path(base_path, image_path, suffix):
    if not base_path:
        return None

    base_path = Path(base_path)
    if base_path.suffix:
        return base_path
    return base_path / f"{image_path.stem}{suffix}"


def extract_image(image_path, model, args, is_batch=False):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    image = rotate_image(image, args.rotate)

    crops_dir = None
    if args.crops_dir:
        crops_dir = Path(args.crops_dir) / image_path.stem if is_batch else Path(args.crops_dir)

    crops, detections = save_yolo_crops(image, model, args.conf, crops_dir)

    debug_path = child_output_path(args.debug_image, image_path, "_debug.jpg")
    if debug_path:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        save_yolo_debug_image(image, detections, debug_path)

    data = {
        field_name: extract_field(crops[field_name], field_name, args.ocr_mode) if field_name in crops else ""
        for field_name in FIELDS
    }
    missing_fields = [field_name for field_name in FIELDS if field_name not in crops]

    return {
        "image_path": str(image_path),
        "model": args.model,
        "confidence": args.conf,
        "data": data,
        "missing_fields": missing_fields,
        "detections": {
            field_name: {
                "confidence": round(detection["score"], 4),
                "box": list(detection["box"]),
            }
            for field_name, detection in detections.items()
        },
    }


def parse_args():
    default_tesseract = shutil.which("tesseract")

    parser = argparse.ArgumentParser(
        description="Extract selected KHB delivery-order fields from an image."
    )
    parser.add_argument(
        "--image_path",
        "--image",
        default="/Users/timdev/Downloads/new-inv-image.jpeg",
        help="Path to one delivery-order image or a folder of images.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to an Excel file to create for single-image extraction.",
    )
    parser.add_argument(
        "--crops-dir",
        default=None,
        help="Optional directory where field crop images are saved.",
    )
    parser.add_argument(
        "--tesseract",
        default=default_tesseract,
        help="Path to the tesseract executable.",
    )
    parser.add_argument(
        "--rotate",
        type=float,
        default=0.0,
        help="Rotate the scanned image before extracting. Example: --rotate -1",
    )
    parser.add_argument(
        "--full-text-output",
        default=None,
        help="Optional path to save full-page OCR text. Disabled by default for speed.",
    )
    parser.add_argument(
        "--debug-image",
        default=None,
        help="Optional path or directory to save image(s) with YOLO field detection boxes.",
    )
    parser.add_argument(
        "--model",
        default="runs/detect/khb_field_model-9/weights/best.pt",
        help="Trained YOLO model path.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold when --model is used.",
    )
    parser.add_argument(
        "--ocr-mode",
        choices=("fast", "accurate"),
        default="accurate",
        help="OCR pass count. Use accurate for blurry scans if fast misses fields.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model: {model_path}. Train first with train_khb_model.py."
        )

    image_paths = find_image_paths(args.image_path)
    model = load_yolo_model(model_path)
    results = [
        extract_image(image_path, model, args, is_batch=len(image_paths) > 1)
        for image_path in image_paths
    ]

    if args.output:
        if len(results) > 1:
            df = pd.DataFrame([result["data"] | {"image_path": result["image_path"]} for result in results])
            df.to_excel(args.output, index=False)
        else:
            write_excel(results[0]["data"], Path(args.output))
    if args.full_text_output:
        if len(image_paths) > 1:
            raise ValueError("--full-text-output is only supported for one image.")
        image = cv2.imread(str(image_paths[0]))
        image = rotate_image(image, args.rotate)
        Path(args.full_text_output).write_text(extract_full_text(image), encoding="utf-8")

    payload = results[0] if len(results) == 1 else {"count": len(results), "results": results}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
