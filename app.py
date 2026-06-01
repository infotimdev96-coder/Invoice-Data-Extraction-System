import argparse
import contextlib
import io
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

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
# DEFAULT_MODEL_PATH = "runs/detect/khb_field_model_6fields/weights/best.pt"
# DEFAULT_MODEL_PATH = "runs/detect/khb_field_model_batch8/weights/best.pt"
# DEFAULT_MODEL_PATH = "runs/detect/khb_field_model_new_templates/weights/best.pt"

DEFAULT_MODEL_PATH = "my_model/v2_best.pt"
DEFAULT_TESSERACT = shutil.which("tesseract")
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TESSDATA_DIR = Path(
    os.environ.get("OCR_TESSDATA_DIR", PROJECT_DIR / "tools" / "tesstrain" / "data")
)
DEFAULT_OCR_LANG = os.environ.get("OCR_LANG", "khb_invoice_best")
DEFAULT_CROPS_DIR = os.environ.get("OCR_CROPS_DIR") or None
TESSDATA_BEST_DIR = Path(
    os.environ.get(
        "TESSDATA_BEST_DIR",
        PROJECT_DIR / "training" / "tessdata_best_repo"
        if (PROJECT_DIR / "training" / "tessdata_best_repo").exists()
        else PROJECT_DIR / "training" / "tessdata_best",
    )
)
TESSDATA_BEST_LANG = os.environ.get("TESSDATA_BEST_LANG", "eng")
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
OCR_WHITELISTS = {
    "Invoice No": "0123456789",
    "Invoice Date": "0123456789./-",
    "Dealer Code": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "Sale Order": "0123456789",
    "Vender Code": "0123456789",
    "Vehicle Code": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
}

if DEFAULT_TESSERACT:
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT

def load_yolo_model(model_path):
    """Load a YOLO model without printing library progress output."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return YOLO(str(model_path))


@lru_cache(maxsize=4)
def get_cached_model(model_path):
    """Load each YOLO checkpoint only once per API process."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return load_yolo_model(path)


def warm_default_model():
    """Load model weights and initialize prediction before the first request."""
    if os.environ.get("WARM_YOLO_MODEL", "1") == "0":
        return
    try:
        model = get_cached_model(DEFAULT_MODEL_PATH)
        model.predict(np.zeros((64, 64, 3), dtype=np.uint8), imgsz=64, verbose=False)
    except FileNotFoundError:
        pass


@asynccontextmanager
async def lifespan(_app):
    """Warm the default model when the API process starts."""
    warm_default_model()
    yield


app = FastAPI(
    title="Invoice Data Extraction API",
    description="Extract invoice values from YOLO field crops with one fast OCR pass per field.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def rotate_image(image, angle):
    """Rotate an image while retaining its original dimensions."""
    if angle == 0:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def decode_uploaded_image(file_bytes):
    """Decode uploaded bytes into a BGR OpenCV image."""
    image = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode uploaded image.")
    return image


def detect_fields(image, model, confidence):
    """Keep the highest-confidence YOLO detection for each invoice field."""
    results = model.predict(image, conf=confidence, verbose=False)
    names = model.names
    best_by_field = {}

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        field_name = names[class_id]
        if field_name not in FIELD_SET:
            continue

        xmin, ymin, xmax, ymax = [int(value) for value in box.xyxy[0].tolist()]
        crop = image[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            continue

        score = float(box.conf[0])
        detection = {
            "field_name": field_name,
            "confidence": score,
            "bbox": [xmin, ymin, xmax, ymax],
            "crop": crop,
        }
        if field_name not in best_by_field or score > best_by_field[field_name]["confidence"]:
            best_by_field[field_name] = detection

    return best_by_field


def save_crops(detections, output_dir, image_stem):
    """Save YOLO crops and return a crop path for each detected field."""
    crop_dir = Path(output_dir) / image_stem
    crop_dir.mkdir(parents=True, exist_ok=True)

    crop_paths = {}
    for field_name in FIELDS:
        detection = detections.get(field_name)
        if not detection:
            continue
        crop_path = crop_dir / f"{field_name.replace(' ', '_')}.png"
        if not cv2.imwrite(str(crop_path), detection["crop"]):
            raise OSError(f"Could not save crop image: {crop_path}")
        crop_paths[field_name] = crop_path
    return crop_paths


def convert_crop_for_text(crop):
    """Isolate blue value text from a complete YOLO crop for Tesseract."""
    if crop is None or crop.size == 0:
        raise ValueError("Cannot convert an empty crop.")

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    value_mask = cv2.inRange(
        hsv,
        np.array([100, 40, 30], dtype=np.uint8),
        np.array([135, 255, 255], dtype=np.uint8),
    )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(value_mask) >= 10:
        x, y, width, height = cv2.boundingRect(value_mask)
        padding = 8
        x_start = max(x - padding, 0)
        y_start = max(y - padding, 0)
        x_end = min(x + width + padding, gray.shape[1])
        y_end = min(y + height + padding, gray.shape[0])
        gray = gray[y_start:y_end, x_start:x_end]

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, converted = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    scale = min(4.0, max(1.0, 80.0 / max(converted.shape[0], 1)))
    if scale > 1.0:
        converted = cv2.resize(converted, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return cv2.copyMakeBorder(converted, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)


def read_crop_image(crop_path):
    """Load a complete saved YOLO crop path and prepare it for OCR."""
    crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if crop is None:
        raise FileNotFoundError(f"Could not load crop image: {crop_path}")
    return convert_crop_for_text(crop)


def clean_text(field_name, value):
    """Normalize one OCR value for the public response."""
    value = value.strip()
    if field_name in {"Invoice No", "Sale Order", "Vender Code"}:
        digits = re.sub(r"\D", "", value)
        if field_name == "Sale Order" and digits:
            return digits.zfill(10)
        return digits
    if field_name == "Invoice Date":
        match = re.search(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", value)
        if match:
            day, month, year = match.groups()
            if int(month) > 12 and month.startswith("8"):
                month = f"0{month[-1]}"
            return f"{int(day):02d}.{int(month):02d}.{year}"
        return value.replace(" ", "")
    if field_name == "Dealer Code":
        return re.sub(r"[^A-Za-z0-9]", "", value).upper()
    if field_name == "Vehicle Code":
        cleaned = re.sub(r"[^A-Za-z0-9-]", "", value).upper()
        missing_prefix_letter = re.fullmatch(r"(\d)-+(\d{4})", cleaned)
        if missing_prefix_letter:
            return f"{missing_prefix_letter.group(1)}G-{missing_prefix_letter.group(2)}"
        if "-" in cleaned:
            prefix, suffix = cleaned.split("-", 1)
            suffix = suffix.translate(
                str.maketrans({"O": "0", "S": "5", "I": "1", "L": "1", "B": "8", "R": "8"})
            )
            return f"{prefix}-{suffix}"
        return cleaned
    return value


def ocr_image(image, field_name, tessdata_dir, lang):
    """Read one crop image with a selected Tesseract model."""
    config = (
        f'--tessdata-dir "{tessdata_dir}" '
        f"-l {lang} "
        "--oem 1 "
        "--psm 7 "
        f"-c tessedit_char_whitelist={OCR_WHITELISTS[field_name]}"
    )
    raw_text = pytesseract.image_to_string(image, config=config)
    return {
        "lang": lang,
        "tessdata_dir": str(tessdata_dir),
        "raw": raw_text,
        "text": clean_text(field_name, raw_text),
    }


def score_ocr_candidate(field_name, text):
    """Score cleaned OCR text so the clearest candidate wins."""
    if not text:
        return 0
    if field_name == "Invoice Date":
        try:
            datetime.strptime(text, "%d.%m.%Y")
        except ValueError:
            return len(text)
        return 100
    if field_name == "Vehicle Code":
        return 100 + len(text) if re.fullmatch(r"[A-Z0-9]+-\d+", text) else len(text)
    if field_name == "Dealer Code":
        return 100 + len(text) if re.fullmatch(r"[A-Z0-9]+", text) else len(text)
    if field_name in {"Invoice No", "Sale Order", "Vender Code"}:
        return 100 + len(text) if text.isdigit() else len(text)
    return len(text)


def extract_text_from_crop(crop_path, field_name):
    """Read text from a saved YOLO crop path and choose the clearest OCR pass."""
    image = read_crop_image(crop_path)
    candidates = []

    if (DEFAULT_TESSDATA_DIR / f"{DEFAULT_OCR_LANG}.traineddata").exists():
        candidates.append(ocr_image(image, field_name, DEFAULT_TESSDATA_DIR, DEFAULT_OCR_LANG))

    if (TESSDATA_BEST_DIR / f"{TESSDATA_BEST_LANG}.traineddata").exists():
        candidates.append(ocr_image(image, field_name, TESSDATA_BEST_DIR, TESSDATA_BEST_LANG))

    if not candidates:
        raise FileNotFoundError(
            "No Tesseract OCR model found. Expected "
            f"{DEFAULT_TESSDATA_DIR / f'{DEFAULT_OCR_LANG}.traineddata'} or "
            f"{TESSDATA_BEST_DIR / f'{TESSDATA_BEST_LANG}.traineddata'}."
        )

    best = max(candidates, key=lambda candidate: score_ocr_candidate(field_name, candidate["text"]))
    return best["text"]


def extract_invoice_image(
    image,
    filename="uploaded_invoice",
    model_path=DEFAULT_MODEL_PATH,
    conf=0.25,
    rotate=0.0,
    crops_dir=None,
):
    """Run YOLO, OCR temporary saved crop paths, and remove them by default."""
    image = rotate_image(image, rotate)
    model = get_cached_model(str(model_path))
    detections = detect_fields(image, model, conf)

    base_stem = Path(filename).stem or "uploaded_invoice"
    image_stem = f"{base_stem}_{uuid4().hex[:8]}"
    temporary_crops = None
    if crops_dir is None:
        temporary_crops = tempfile.TemporaryDirectory(prefix="invoice-crops-")
        crops_dir = temporary_crops.name

    try:
        crop_paths = save_crops(detections, crops_dir, image_stem)
        data = {}
        missing_fields = []
        for field_name in FIELDS:
            crop_path = crop_paths.get(field_name)
            if not crop_path:
                data[field_name] = ""
                missing_fields.append(field_name)
                continue
            try:
                data[field_name] = extract_text_from_crop(crop_path, field_name)
            except Exception:
                data[field_name] = ""
            if not data[field_name]:
                missing_fields.append(field_name)

        return {
            "filename": filename,
            "data": data,
            "missing_fields": missing_fields,
        }
    finally:
        if temporary_crops:
            temporary_crops.cleanup()


def format_api_data(data):
    """Map internal field names to the API response contract."""
    formatted = {
        api_key: data.get(field_name, "")
        for field_name, api_key in API_FIELD_MAP.items()
    }
    formatted["warehouse"] = DEFAULT_WAREHOUSE
    formatted["route"] = DEFAULT_ROUTE
    return formatted


def build_api_response(result):
    """Build the public JSON response shape."""
    return {
        "success": True,
        "result": {
            "data": format_api_data(result["data"]),
            "missingFields": [API_FIELD_MAP.get(field, field) for field in result["missing_fields"]],
            "filename": result["filename"],
        },
    }


@app.get("/")
def api_root():
    """Return basic API usage information."""
    return {
        "message": "Invoice Data Extraction API",
        "upload_endpoint": "/api/v1/ocr/single",
        "supported_extensions": sorted(IMAGE_EXTENSIONS),
    }


@app.get("/health")
def api_health(model: str = Query(DEFAULT_MODEL_PATH, description="YOLO model path to check.")):
    """Report runtime dependency and model availability."""
    return {
        "status": "ok",
        "model": model,
        "model_exists": Path(model).exists(),
        "tesseract": DEFAULT_TESSERACT,
        "ocr_lang": DEFAULT_OCR_LANG,
        "tessdata_dir": str(DEFAULT_TESSDATA_DIR),
        "crops_dir": DEFAULT_CROPS_DIR,
        "ocr_model_exists": (DEFAULT_TESSDATA_DIR / f"{DEFAULT_OCR_LANG}.traineddata").exists(),
        "tessdata_best_lang": TESSDATA_BEST_LANG,
        "tessdata_best_dir": str(TESSDATA_BEST_DIR),
        "tessdata_best_model_exists": (TESSDATA_BEST_DIR / f"{TESSDATA_BEST_LANG}.traineddata").exists(),
    }


@app.post("/api/v1/ocr/single")
async def api_extract_invoice(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL_PATH, description="YOLO model path."),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="YOLO confidence threshold."),
    rotate: float = Query(0.0, description="Degrees to rotate the image before extraction."),
    crops_dir: str | None = Query(DEFAULT_CROPS_DIR, description="Directory to keep saved YOLO crops."),
):
    """Upload one invoice and extract text from its saved YOLO crop paths."""
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
            rotate=rotate,
            crops_dir=crops_dir,
        )
        return build_api_response(result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc
    finally:
        await file.close()


def find_image_paths(image_path):
    """Find all supported images in a file or directory path."""
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


def extract_invoice_file(
    image_path,
    model_path=DEFAULT_MODEL_PATH,
    conf=0.25,
    rotate=0.0,
    crops_dir=None,
):
    """Extract one invoice image from disk."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return extract_invoice_image(
        image,
        filename=Path(image_path).name,
        model_path=model_path,
        conf=conf,
        rotate=rotate,
        crops_dir=crops_dir,
    )


def parse_args():
    """Parse command-line options for local batch extraction."""
    parser = argparse.ArgumentParser(description="Extract invoice values from saved YOLO crop paths")
    parser.add_argument("--image_path", "--image", default="images", help="Path to image or folder")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--rotate", type=float, default=0.0, help="Rotate image before extraction")
    parser.add_argument(
        "--crops-dir",
        default=None,
        help="Optional directory to keep saved YOLO crops. By default crops are deleted after OCR.",
    )
    parser.add_argument("--output", default=None, help="Optional Excel output path")
    return parser.parse_args()


def main():
    """Extract one image or a folder of images."""
    args = parse_args()
    results = [
        extract_invoice_file(
            image_path,
            model_path=args.model,
            conf=args.conf,
            rotate=args.rotate,
            crops_dir=args.crops_dir,
        )
        for image_path in find_image_paths(args.image_path)
    ]

    if args.output:
        rows = [{"filename": result["filename"], **result["data"]} for result in results]
        pd.DataFrame(rows).to_excel(args.output, index=False)
        print(f"Excel saved: {args.output}")

    payload = results[0] if len(results) == 1 else {"count": len(results), "results": results}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
