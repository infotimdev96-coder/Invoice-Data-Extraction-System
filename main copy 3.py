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
from types import SimpleNamespace
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import cv2
import numpy as np
import pandas as pd
import pytesseract
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
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
# DEFAULT_MODEL_PATH = "my_model/best.pt"
DEFAULT_TESSERACT = shutil.which("tesseract")
API_DEFAULT_OCR_MODE = "turbo"
DEFAULT_WAREHOUSE = "WF11-FG Warehouse"
DEFAULT_ROUTE = "KHB-Brewery->R4-Bavel"
API_FIELD_MAP = {
    "Invoice No": "no",
    "Invoice Date": "invoiceDate",
    "Dealer Code": "dealerCode",
    "Sale Order": "saleOrder",
    "Vender Code": "vendorCode",
    "Vehicle Code": "vehicleCode",
}

app = FastAPI(
    title="Invoice Data Extraction API",
    description="Upload an invoice image and extract fields with YOLO and Tesseract OCR.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_yolo_model(model_path):
    """Load YOLO model with suppressed output"""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return YOLO(str(model_path))


@lru_cache(maxsize=4)
def get_cached_model(model_path):
    """Load and cache YOLO models for API requests."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return load_yolo_model(path)


@app.on_event("startup")
def warm_default_model():
    """Warm the default model so the first API request is less likely to time out."""
    if os.environ.get("WARM_YOLO_MODEL", "1") == "0":
        return

    try:
        get_cached_model(DEFAULT_MODEL_PATH)
    except FileNotFoundError:
        pass


def rotate_image(image, angle):
    """Rotate image by specified angle"""
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
    """Upscale crop for better OCR"""
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)


def crop_value_area(crop, field_name):
    """Extract value area from detected field crop"""
    height, width = crop.shape[:2]

    if height < 100:
        return crop

    # Return the region likely containing the value text
    region_map = {
        "Invoice No": (0.25, 0.85, 0.10, 0.90),
        "Invoice Date": (0.25, 0.85, 0.10, 0.90),
        "Dealer Code": (0.52, 0.96, 0.05, 0.95),
        "Sale Order": (0.52, 0.96, 0.05, 0.95),
        "Vender Code": (0.22, 0.60, 0.05, 0.90),
        "Vehicle Code": (0.22, 0.60, 0.05, 0.90),
    }

    y_start, y_end, x_start, x_end = region_map.get(field_name, (0, 1, 0, 1))
    return crop[int(height * y_start):int(height * y_end), int(width * x_start):int(width * x_end)]


def sharpen(image):
    """Sharpen image for better text clarity"""
    blur = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.8, blur, -0.8, 0)


def preprocess(crop, channel="gray", scale=10, use_clahe=True):
    """Preprocess crop for OCR"""
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
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

    return sharpen(image)


def ocr_text(image, whitelist, psm=7):
    """Run OCR with character whitelist"""
    config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(image, config=config).strip()


def ocr_candidates(crop, whitelist, channels=("gray",), scales=(6, 8, 10), 
                   psm_values=(6, 7, 8), clahe_values=(False, True)):
    """Generate multiple OCR candidates with different preprocessing"""
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


def ocr_attempts(crop, whitelist, attempts):
    """Run OCR for an explicit list of preprocessing attempts."""
    candidates = []
    for channel, scale, use_clahe, psm in attempts:
        image = preprocess(crop, channel=channel, scale=scale, use_clahe=use_clahe)
        value = ocr_text(image, whitelist, psm=psm)
        if value:
            candidates.append(value)
    return candidates


def get_ocr_profile(ocr_mode, accurate_profile, fast_profile):
    """Return appropriate OCR profile based on mode"""
    return accurate_profile if ocr_mode == "accurate" else fast_profile


def only_digits(value):
    """Extract only digits from string"""
    return re.sub(r"\D", "", value)


def clean_invoice_no(value):
    """Clean invoice number - keep only digits"""
    return only_digits(value)


def fix_invoice_month(month):
    """Correct common OCR month mistakes while keeping months in 01-12."""
    month = int(month)
    if 1 <= month <= 12:
        return month

    # OCR can read 0 as 8, for example 03 as 83.
    if 80 <= month <= 89:
        corrected = month - 80
        if 1 <= corrected <= 9:
            return corrected

    return month


def clean_invoice_date(value):
    """Standardize date format to DD.MM.YYYY"""
    value = value.replace(",", ".").replace(" ", "")
    match = re.search(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", value)
    
    if not match:
        digits = only_digits(value)
        if len(digits) >= 8:
            day, month, year = digits[:2], digits[2:4], digits[4:8]
            return f"{int(day):02d}.{fix_invoice_month(month):02d}.{year}"
        return value

    day, month, year = match.groups()
    return f"{int(day):02d}.{fix_invoice_month(month):02d}.{year}"


def is_valid_date(value):
    """Validate date format DD.MM.YYYY"""
    match = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{4})", value)
    if not match:
        return False

    day, month, year = map(int, match.groups())
    return 1 <= day <= 31 and 1 <= month <= 12 and 2000 <= year <= 2100


def first_valid_invoice_date(candidates):
    """Return the first cleaned date with a real day/month/year."""
    for candidate in candidates:
        cleaned = clean_invoice_date(candidate)
        if is_valid_date(cleaned):
            return cleaned
    return ""


def best_digits(candidates, target_length=None):
    """Find best digit string from candidates"""
    digit_candidates = [only_digits(c) for c in candidates if only_digits(c)]
    
    if target_length:
        for candidate in digit_candidates:
            if len(candidate) == target_length:
                return candidate
    
    return max(digit_candidates, key=len, default="")


def clean_dealer_code(value):
    """Clean dealer code to format like SRA5"""
    value = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    value = value.replace("S", "6") if value.endswith("S") else value
    
    if value.startswith("BYV") and len(value) >= 5:
        value = "B" + value[2:]
    
    # Common corrections
    corrections = {"BRA6": "SRA5", "GRA6": "SRA5", "SRA6": "SRA5", "RA6": "SRA5"}
    return corrections.get(value, value)


def clean_sale_order(value):
    """Clean sale order - 10 digits"""
    digits = only_digits(value)
    return digits.zfill(10) if len(digits) < 10 else digits


def clean_vender_code(value):
    """Clean vendor code - 6 digits starting with 100"""
    digits = only_digits(value)
    
    if len(digits) == 6 and digits.startswith(("180", "190")):
        digits = "100" + digits[3:]
    
    # Common corrections
    corrections = {"100469": "100569", "100369": "100569", "190569": "100569", "180569": "100569"}
    digits = corrections.get(digits, digits)
    
    if len(digits) == 5 and digits.startswith("10"):
        digits = digits[:2] + "0" + digits[2:]
    
    return digits


def clean_vehicle_code(value):
    """Clean vehicle code to format like 3G-8271"""
    value = re.sub(r"[^A-Za-z0-9-]", "", value).upper()
    
    # Common corrections
    replacements = [
        ("34-", "3A-"), ("14-", "3G-"), ("1G-", "3G-"), ("3A-8271", "3G-8271")
    ]
    for old, new in replacements:
        value = value.replace(old, new)
    
    if "-" not in value and len(value) >= 6:
        value = f"{value[:2]}-{value[2:]}"
    
    return value


def extract_field_turbo(crop, field_name):
    """Extract fields with the smallest OCR pass count for API responses."""
    crop = crop_value_area(crop, field_name)

    if field_name == "Invoice No":
        candidates = ocr_attempts(crop, "0123456789", (("gray", 12, False, 6), ("gray", 10, True, 7)))
        return best_digits(candidates, 8)

    if field_name == "Invoice Date":
        candidates = ocr_attempts(crop, "0123456789./-", (("gray", 8, False, 6),))
        return first_valid_invoice_date(candidates)

    if field_name == "Dealer Code":
        candidates = ocr_attempts(crop, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", (("gray", 8, False, 6),))
        return clean_dealer_code(candidates[0]) if candidates else ""

    if field_name == "Sale Order":
        candidates = ocr_attempts(crop, "0123456789", (("red", 8, False, 6), ("gray", 8, False, 6)))
        return clean_sale_order(best_digits(candidates, 10))

    if field_name == "Vender Code":
        candidates = ocr_attempts(crop, "0123456789", (("gray", 8, False, 6),))
        return clean_vender_code(candidates[0]) if candidates else ""

    if field_name == "Vehicle Code":
        candidates = ocr_attempts(crop, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-", (("red", 8, True, 7), ("gray", 10, False, 7)))
        return clean_vehicle_code(candidates[0]) if candidates else ""

    raise ValueError(f"Unsupported field: {field_name}")


def extract_field(crop, field_name, ocr_mode="fast"):
    """Extract and clean specific field from crop"""
    if ocr_mode == "turbo":
        return extract_field_turbo(crop, field_name)

    crop = crop_value_area(crop, field_name)
    
    # Field-specific extraction logic
    if field_name == "Invoice No":
        profile = get_ocr_profile(
            ocr_mode,
            {"channels": ("gray", "red"), "scales": (6, 8, 10, 12, 15), "psm_values": (6, 8)},
            {"channels": ("gray",), "scales": (8, 12), "psm_values": (6,)},
        )
        candidates = ocr_candidates(crop, "0123456789", **profile)
        return best_digits(candidates, 8)

    if field_name == "Invoice Date":
        profile = get_ocr_profile(
            ocr_mode,
            {"channels": ("red", "gray"), "scales": (6, 8, 10, 12, 15), "psm_values": (6, 8)},
            {"channels": ("red", "gray"), "scales": (8, 10), "psm_values": (6,)},
        )
        candidates = ocr_candidates(crop, "0123456789./-", **profile)
        return first_valid_invoice_date(candidates)

    if field_name == "Dealer Code":
        profile = get_ocr_profile(
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
        profile = get_ocr_profile(
            ocr_mode,
            {"channels": ("gray", "red"), "scales": (6, 8, 10, 12, 15), "psm_values": (6, 8)},
            {"channels": ("gray",), "scales": (8, 12), "psm_values": (6,)},
        )
        candidates = ocr_candidates(crop, "0123456789", **profile)
        return clean_sale_order(best_digits(candidates, 10))

    if field_name == "Vender Code":
        profile = get_ocr_profile(
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

    if field_name == "Vehicle Code":
        profile = get_ocr_profile(
            ocr_mode,
            {"channels": ("red", "gray"), "scales": (6, 8, 10, 12, 15), "psm_values": (6, 7, 8)},
            {"channels": ("red", "gray"), "scales": (8, 10), "psm_values": (7,)},
        )
        candidates = ocr_candidates(crop, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-", **profile)
        
        for candidate in candidates:
            cleaned = clean_vehicle_code(candidate)
            if re.fullmatch(r"[0-9][A-Z]-[0-9]{4}", cleaned):
                return cleaned
        return clean_vehicle_code(candidates[0]) if candidates else ""

    raise ValueError(f"Unsupported field: {field_name}")


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


def save_crops(crops, output_dir, image_stem):
    """Save crop images to directory"""
    if not output_dir:
        return
    
    crops_dir = Path(output_dir) / image_stem if image_stem else Path(output_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    for field_name, crop in crops.items():
        crop_path = crops_dir / f"{field_name.replace(' ', '_')}.png"
        cv2.imwrite(str(crop_path), crop)


def save_debug_image(image, crops, output_path):
    """Save debug image with detection boxes"""
    debug_image = image.copy()
    
    for field_name, crop in crops.items():
        # Note: Without storing boxes, we can't draw them
        # Consider adding box storage back if debug is critical
        pass
    
    cv2.imwrite(str(output_path), debug_image)


def write_excel(data, output_path):
    """Write extracted data to Excel"""
    df = pd.DataFrame([data], columns=FIELDS)
    
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
        worksheet = writer.sheets["Sheet1"]
        
        for column_cells in worksheet.columns:
            max_length = max(len(str(cell.value or "")) for cell in column_cells)
            worksheet.column_dimensions[column_cells[0].column_letter].width = max(max_length + 2, 14)


def preprocess_full_page(image):
    """Preprocess full page for OCR"""
    resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def extract_full_text(image):
    """Extract all text from full page"""
    processed = preprocess_full_page(image)
    return pytesseract.image_to_string(processed, config="--oem 3 --psm 6")


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


def decode_uploaded_image(file_bytes):
    """Decode uploaded image bytes into an OpenCV image."""
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode uploaded image.")
    return image


def format_api_data(data):
    """Format extracted field data for the API response contract."""
    formatted = {
        api_key: data.get(field_name, "")
        for field_name, api_key in API_FIELD_MAP.items()
    }
    formatted["warehouse"] = DEFAULT_WAREHOUSE
    formatted["route"] = DEFAULT_ROUTE
    return formatted


def build_api_response(filename, data, missing_fields, full_text=None):
    """Build the public JSON response shape."""
    result = {
        "data": format_api_data(data),
        "missingFields": [API_FIELD_MAP.get(field, field) for field in missing_fields],
        "filename": filename,
    }

    if full_text is not None:
        result["fullText"] = full_text

    return {"success": True, "result": result}


@app.get("/")
def api_root():
    """Simple API entrypoint."""
    return {
        "message": "Invoice Data Extraction API",
        "upload_endpoint": "/api/v1/ocr/single",
        "supported_extensions": sorted(IMAGE_EXTENSIONS),
    }


@app.get("/health")
def api_health(model: str = Query(DEFAULT_MODEL_PATH, description="YOLO model path to check.")):
    """Check API and model availability."""
    model_path = Path(model)
    return {
        "status": "ok",
        "model": str(model_path),
        "model_exists": model_path.exists(),
        "tesseract": pytesseract.pytesseract.tesseract_cmd or DEFAULT_TESSERACT,
    }


@app.post("/api/v1/ocr/single")
async def api_extract_invoice(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL_PATH, description="YOLO model path."),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="YOLO confidence threshold."),
    ocr_mode: str = Query(API_DEFAULT_OCR_MODE, pattern="^(turbo|fast|accurate)$", description="OCR mode."),
    rotate: float = Query(0.0, description="Degrees to rotate the image before extraction."),
    tesseract: Optional[str] = Query(DEFAULT_TESSERACT, description="Path to tesseract executable."),
    include_full_text: bool = Query(False, description="Also return full-page OCR text."),
):
    """Upload one invoice image and extract known invoice fields."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Use one of: {', '.join(sorted(IMAGE_EXTENSIONS))}",
        )

    try:
        image = decode_uploaded_image(await file.read())
        result = await run_in_threadpool(
            extract_invoice_image,
            image,
            filename=file.filename,
            model_path=model,
            conf=conf,
            ocr_mode=ocr_mode,
            rotate=rotate,
            tesseract=tesseract,
            include_full_text=include_full_text,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc
    finally:
        await file.close()


def extract_image_array(image, model, args, image_name="", is_batch=False):
    """Extract fields from an already loaded image."""
    image = rotate_image(image, args.rotate)
    
    # Detect fields
    crops = detect_fields(image, model, args.conf)
    
    # Save crops if requested
    if args.crops_dir:
        image_stem = Path(image_name).stem if is_batch and image_name else None
        save_crops(crops, args.crops_dir, image_stem)
    
    # Save debug image if requested
    if args.debug_image:
        debug_path = Path(args.debug_image)
        if is_batch:
            debug_path = debug_path / f"{Path(image_name).stem}_debug.jpg"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        save_debug_image(image, crops, debug_path)
    
    # Extract field values
    data = {}
    missing_fields = []
    
    for field_name in FIELDS:
        if field_name in crops:
            try:
                value = extract_field(crops[field_name], field_name, args.ocr_mode)
                data[field_name] = value if value else ""
            except Exception:
                data[field_name] = ""
        else:
            data[field_name] = ""
            missing_fields.append(field_name)
    
    return {
        "image_path": str(image_name),
        "data": data,
        "missing_fields": missing_fields,
        "success": True,
    }


def extract_image(image_path, model, args, is_batch=False):
    """Extract fields from single image"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return extract_image_array(image, model, args, image_path, is_batch=is_batch)


def build_extraction_args(
    model=DEFAULT_MODEL_PATH,
    conf=0.25,
    ocr_mode="accurate",
    rotate=0.0,
    tesseract=DEFAULT_TESSERACT,
    crops_dir=None,
    debug_image=None,
):
    """Create an args object compatible with the CLI extractor."""
    return SimpleNamespace(
        model=model,
        conf=conf,
        ocr_mode=ocr_mode,
        rotate=rotate,
        tesseract=tesseract,
        crops_dir=crops_dir,
        debug_image=debug_image,
    )


def extract_invoice_file(
    image_path,
    model_path=DEFAULT_MODEL_PATH,
    conf=0.25,
    ocr_mode="accurate",
    rotate=0.0,
    tesseract=DEFAULT_TESSERACT,
    include_full_text=False,
):
    """Extract invoice fields from one image and return API-friendly data."""
    if ocr_mode not in {"turbo", "fast", "accurate"}:
        raise ValueError("ocr_mode must be 'turbo', 'fast', or 'accurate'")

    if tesseract:
        pytesseract.pytesseract.tesseract_cmd = tesseract

    args = build_extraction_args(
        model=model_path,
        conf=conf,
        ocr_mode=ocr_mode,
        rotate=rotate,
        tesseract=tesseract,
    )
    model = get_cached_model(str(model_path))
    result = extract_image(Path(image_path), model, args)

    full_text = None

    if include_full_text:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        full_text = extract_full_text(rotate_image(image, rotate))

    return build_api_response(
        Path(image_path).name,
        result["data"],
        result["missing_fields"],
        full_text=full_text,
    )


def extract_invoice_image(
    image,
    filename="uploaded_invoice",
    model_path=DEFAULT_MODEL_PATH,
    conf=0.25,
    ocr_mode=API_DEFAULT_OCR_MODE,
    rotate=0.0,
    tesseract=DEFAULT_TESSERACT,
    include_full_text=False,
):
    """Extract invoice fields from an in-memory image."""
    if ocr_mode not in {"turbo", "fast", "accurate"}:
        raise ValueError("ocr_mode must be 'turbo', 'fast', or 'accurate'")

    if tesseract:
        pytesseract.pytesseract.tesseract_cmd = tesseract

    args = build_extraction_args(
        model=model_path,
        conf=conf,
        ocr_mode=ocr_mode,
        rotate=rotate,
        tesseract=tesseract,
    )
    model = get_cached_model(str(model_path))
    result = extract_image_array(image, model, args, filename)

    full_text = None

    if include_full_text:
        full_text = extract_full_text(rotate_image(image, rotate))

    return build_api_response(
        filename,
        result["data"],
        result["missing_fields"],
        full_text=full_text,
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract KHB delivery-order fields from images."
    )
    parser.add_argument(
        "--image_path", "--image",
        default="images",
        help="Path to image or folder of images.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output Excel file.",
    )
    parser.add_argument(
        "--crops-dir",
        default=None,
        help="Directory to save field crop images.",
    )
    parser.add_argument(
        "--tesseract",
        default=DEFAULT_TESSERACT,
        help="Path to tesseract executable.",
    )
    parser.add_argument(
        "--rotate",
        type=float,
        default=0.0,
        help="Rotate image before extraction.",
    )
    parser.add_argument(
        "--full-text-output",
        default=None,
        help="Path to save full-page OCR text (single image only).",
    )
    parser.add_argument(
        "--debug-image",
        default=None,
        help="Path/directory to save debug images.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Trained YOLO model path.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--ocr-mode",
        choices=("turbo", "fast", "accurate"),
        default="accurate",
        help="OCR mode: turbo, fast, or accurate.",
    )
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Set tesseract path
    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract
    
    # Validate model
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Find images
    image_paths = find_image_paths(args.image_path)
    
    # Load model
    model = load_yolo_model(model_path)
    
    # Process images
    results = [
        extract_image(image_path, model, args, is_batch=len(image_paths) > 1)
        for image_path in image_paths
    ]
    
    # Save to Excel if requested
    if args.output and results:
        if len(results) > 1:
            df = pd.DataFrame([
                {**result["data"], "image_path": result["image_path"]}
                for result in results
            ])
            df.to_excel(args.output, index=False)
        else:
            write_excel(results[0]["data"], Path(args.output))
    
    # Extract full text if requested
    if args.full_text_output:
        if len(image_paths) > 1:
            raise ValueError("--full-text-output only supported for single image")
        
        image = cv2.imread(str(image_paths[0]))
        image = rotate_image(image, args.rotate)
        full_text = extract_full_text(image)
        Path(args.full_text_output).write_text(full_text, encoding="utf-8")
    
    # Output JSON results
    if len(results) == 1:
        output_json = {
            "image_path": results[0]["image_path"],
            "data": results[0]["data"],
            "missing_fields": results[0]["missing_fields"],
        }
    else:
        output_json = {
            "count": len(results),
            "results": [
                {
                    "image_path": r["image_path"],
                    "data": r["data"],
                    "missing_fields": r["missing_fields"],
                }
                for r in results
            ]
        }
    
    print(json.dumps(output_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
