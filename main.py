import argparse
import contextlib
import io
import json
import os
import re
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import cv2
import numpy as np
import pytesseract
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
DEFAULT_TESSERACT = shutil.which("tesseract")


def load_yolo_model(model_path):
    """Load YOLO model with suppressed output"""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return YOLO(str(model_path))


@lru_cache(maxsize=4)
def get_cached_model(model_path):
    """Load and cache YOLO models"""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return load_yolo_model(path)


def rotate_image(image, angle):
    """Rotate image by specified angle"""
    if angle == 0:
        return image
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def crop_value_region_for_ocr(crop, field_name=None):
    """Crop the detected YOLO field down to the likely blue value text region."""
    if crop is None or crop.size == 0:
        return crop

    if len(crop.shape) != 3:
        return crop

    height, width = crop.shape[:2]
    blue, green, red = cv2.split(crop)
    blue_text_mask = (blue > red + 10) & (blue > green + 8) & (blue > 70)

    if field_name in {"Invoice Date", "Invoice No"}:
        cutoff = int(height * 0.10)
    elif field_name in {"Vehicle Code", "Vender Code"}:
        cutoff = int(height * 0.22)
    else:
        cutoff = int(height * 0.42)

    blue_text_mask[:cutoff, :] = False
    ys, xs = np.where(blue_text_mask)

    if len(xs) < 10:
        return crop[cutoff:height, 0:width]

    x_min = max(int(xs.min()) - 10, 0)
    x_max = min(int(xs.max()) + 11, width)
    y_min = max(int(ys.min()) - 7, 0)
    y_max = min(int(ys.max()) + 8, height)
    return crop[y_min:y_max, x_min:x_max]


def preprocess_for_ocr(crop):
    """Preprocess cropped image for better OCR accuracy"""
    if crop is None or crop.size == 0:
        return None
    
    # Convert to grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Upscale for better OCR
    height, width = denoised.shape
    if height < 100 or width < 200:
        scale = max(2, int(400 / min(height, width)))
        denoised = cv2.resize(denoised, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    return denoised


def build_tesseract_config(base_config, ocr_lang=None, tessdata_dir=None):
    parts = [base_config]
    if tessdata_dir:
        parts.append(f'--tessdata-dir "{tessdata_dir}"')
    if ocr_lang:
        parts.append(f"-l {ocr_lang}")
    return " ".join(parts)


def extract_text_from_cropped_image(cropped_image_path, field_name=None, tesseract_cmd=None, ocr_lang=None, tessdata_dir=None, value_only=False):
    """
    Convert cropped image and extract text using OCR
    
    Args:
        cropped_image_path: Path to the cropped image file
        field_name: Optional field name to apply specific OCR configuration
        tesseract_cmd: Path to tesseract executable
    
    Returns:
        Extracted text as string
    """
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    # Load the cropped image
    crop = cv2.imread(str(cropped_image_path))
    if crop is None:
        raise FileNotFoundError(f"Could not load image: {cropped_image_path}")

    if value_only:
        crop = crop_value_region_for_ocr(crop, field_name)
    
    # Preprocess for OCR
    processed = preprocess_for_ocr(crop)
    if processed is None:
        return ""
    
    # Configure OCR based on field type
    if field_name == "Invoice No":
        config = build_tesseract_config(
            "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789",
            ocr_lang,
            tessdata_dir,
        )
        text = pytesseract.image_to_string(processed, config=config).strip()
        return re.sub(r"\D", "", text)  # Keep only digits
    
    elif field_name == "Invoice Date":
        config = build_tesseract_config(
            "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789./-",
            ocr_lang,
            tessdata_dir,
        )
        text = pytesseract.image_to_string(processed, config=config).strip()
        # Extract date pattern
        match = re.search(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", text)
        if match:
            day, month, year = match.groups()
            return f"{int(day):02d}.{int(month):02d}.{year}"
        return text
    
    elif field_name == "Dealer Code":
        config = build_tesseract_config(
            "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            ocr_lang,
            tessdata_dir,
        )
        text = pytesseract.image_to_string(processed, config=config).strip()
        return re.sub(r"[^A-Za-z0-9]", "", text).upper()
    
    elif field_name == "Sale Order":
        config = build_tesseract_config(
            "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789",
            ocr_lang,
            tessdata_dir,
        )
        text = pytesseract.image_to_string(processed, config=config).strip()
        digits = re.sub(r"\D", "", text)
        return digits.zfill(10) if len(digits) < 10 else digits
    
    elif field_name == "Vender Code":
        config = build_tesseract_config(
            "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789",
            ocr_lang,
            tessdata_dir,
        )
        text = pytesseract.image_to_string(processed, config=config).strip()
        return re.sub(r"\D", "", text)
    
    elif field_name == "Vehicle Code":
        config = build_tesseract_config(
            "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
            ocr_lang,
            tessdata_dir,
        )
        text = pytesseract.image_to_string(processed, config=config).strip()
        return re.sub(r"[^A-Za-z0-9-]", "", text).upper()
    
    else:
        # Default OCR for any other field
        config = build_tesseract_config("--oem 3 --psm 7", ocr_lang, tessdata_dir)
        text = pytesseract.image_to_string(processed, config=config).strip()
        return text


def save_cropped_image(crop, output_dir, filename, field_name, index=0):
    """Save cropped field image to directory"""
    if crop is None or crop.size == 0:
        return None
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate safe filename
    safe_field_name = field_name.replace(" ", "_")
    base_name = Path(filename).stem
    crop_path = output_dir / f"{base_name}_{safe_field_name}_{index}.png"
    
    # Save the cropped image
    cv2.imwrite(str(crop_path), crop)
    return crop_path


def detect_fields(image, model, confidence):
    """Detect fields using YOLO model"""
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
        
        if crop.size > 0:
            detections.append({
                "field_name": field_name,
                "score": score,
                "crop": crop,
            })
    
    # Keep only highest confidence detection per field
    best_by_field = {}
    for detection in detections:
        field_name = detection["field_name"]
        if field_name not in best_by_field or detection["score"] > best_by_field[field_name]["score"]:
            best_by_field[field_name] = detection
    
    return {field: det["crop"] for field, det in best_by_field.items()}


def extract_invoice_data(image, model, conf=0.25, rotate_angle=0.0, crop_dir=None, filename=None, extract_text=True, tesseract_cmd=None, ocr_lang=None, tessdata_dir=None, value_only_ocr=False):
    """Main function to extract invoice data from image"""
    image = rotate_image(image, rotate_angle)
    
    # Detect field crops
    crops = detect_fields(image, model, conf)
    
    # Save cropped images if crop_dir is specified
    saved_crops = []
    if crop_dir and filename:
        for idx, (field_name, crop) in enumerate(crops.items()):
            crop_path = save_cropped_image(crop, crop_dir, filename, field_name, idx)
            if crop_path:
                saved_crops.append({
                    "field_name": field_name,
                    "path": str(crop_path)
                })
    
    # Extract text from crops
    data = {}
    missing_fields = []
    
    for field_name in FIELDS:
        if field_name in crops:
            try:
                if extract_text:
                    # Convert crop to temp file and extract text
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        cv2.imwrite(tmp_file.name, crops[field_name])
                        value = extract_text_from_cropped_image(
                            tmp_file.name,
                            field_name,
                            tesseract_cmd,
                            ocr_lang,
                            tessdata_dir,
                            value_only_ocr,
                        )
                        os.unlink(tmp_file.name)
                else:
                    value = ""
                
                data[field_name] = value if value else ""
                if not value:
                    missing_fields.append(field_name)
            except Exception as e:
                data[field_name] = ""
                missing_fields.append(field_name)
        else:
            data[field_name] = ""
            missing_fields.append(field_name)
    
    return data, missing_fields, saved_crops


def extract_from_file(image_path, model_path=DEFAULT_MODEL_PATH, conf=0.25, rotate_angle=0.0, 
                     crop_dir=None, extract_text=True, tesseract_cmd=None, ocr_lang=None, tessdata_dir=None, value_only_ocr=False):
    """Extract invoice data from image file"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    model = get_cached_model(model_path)
    data, missing_fields, saved_crops = extract_invoice_data(
        image,
        model,
        conf,
        rotate_angle,
        crop_dir,
        Path(image_path).name,
        extract_text,
        tesseract_cmd,
        ocr_lang,
        tessdata_dir,
        value_only_ocr,
    )
    
    result = {
        "filename": Path(image_path).name,
        "data": data,
        "missing_fields": missing_fields
    }
    
    if saved_crops:
        result["saved_crops"] = saved_crops
    
    return result


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Extract invoice fields from images using YOLO + OCR")
    parser.add_argument("--image_path", "--image", default="images", help="Path to image or folder")
    parser.add_argument("--output", default=None, help="Path to output Excel file")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--rotate", type=float, default=0.0, help="Rotate image angle")
    parser.add_argument("--tesseract", default=DEFAULT_TESSERACT, help="Tesseract executable path")
    parser.add_argument("--ocr-lang", default=None, help="Tesseract language/model name, e.g. khb_invoice_values")
    parser.add_argument("--tessdata-dir", default=None, help="Folder containing .traineddata files")
    parser.add_argument(
        "--value-only-ocr",
        action="store_true",
        help="Before OCR, crop each YOLO field to the likely blue value text region.",
    )
    parser.add_argument("--crop-dir", default=None, help="Directory to save cropped field images (optional)")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR text extraction (only save crops)")
    return parser.parse_args()


def find_image_paths(image_path):
    """Find all image files in path"""
    path = Path(image_path)
    
    if path.is_file():
        return [path]
    
    if path.is_dir():
        image_paths = [
            child for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not image_paths:
            raise FileNotFoundError(f"No supported images found in: {path}")
        return sorted(image_paths)
    
    raise FileNotFoundError(f"Image path does not exist: {path}")


def main():
    """Main entry point"""
    args = parse_args()
    
    # Set tesseract path
    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract
    
    # Find images
    image_paths = find_image_paths(args.image_path)
    
    # Process all images
    results = []
    for image_path in image_paths:
        try:
            result = extract_from_file(
                image_path, args.model, args.conf, args.rotate, 
                args.crop_dir,
                not args.no_ocr,
                args.tesseract,
                args.ocr_lang,
                args.tessdata_dir,
                args.value_only_ocr,
            )
            results.append(result)
            print(f"Processed: {result['filename']}")
            if not args.no_ocr:
                print(f"  Data: {result['data']}")
                print(f"  Missing: {result['missing_fields']}")
            if args.crop_dir and 'saved_crops' in result:
                print(f"  Saved crops ({len(result['saved_crops'])}):")
                for crop in result['saved_crops']:
                    print(f"    - {crop['field_name']}: {crop['path']}")
            print()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Save to Excel if requested and OCR was performed
    if args.output and results and not args.no_ocr:
        try:
            import pandas as pd
            df = pd.DataFrame([
                {"filename": r["filename"], **r["data"]}
                for r in results
            ])
            df.to_excel(args.output, index=False)
            print(f"Results saved to {args.output}")
        except ImportError:
            print("Warning: pandas not installed, cannot save to Excel")
    
    # Output JSON
    output_json = {
        "count": len(results),
        "results": results
    }
    print(json.dumps(output_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
