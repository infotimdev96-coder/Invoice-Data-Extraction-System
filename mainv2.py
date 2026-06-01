import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import pytesseract

import main as extractor


ORIGINAL_EXTRACTOR_OCR = extractor.extract_text_from_cropped_image
DEFAULT_YOLO_MODEL = "my_model/best.pt"
FALLBACK_YOLO_MODEL = "runs/detect/khb_field_model_6fields/weights/best.pt"
DEFAULT_TESSDATA_DIR = "tesseract_training/output"
DEFAULT_OCR_LANG = "khb_invoice_values"
DEFAULT_TESSERACT = shutil.which("tesseract")


def existing_default_model():
    if Path(DEFAULT_YOLO_MODEL).exists():
        return DEFAULT_YOLO_MODEL
    return FALLBACK_YOLO_MODEL


def existing_default_ocr_lang(tessdata_dir):
    traineddata = Path(tessdata_dir) / f"{DEFAULT_OCR_LANG}.traineddata"
    return DEFAULT_OCR_LANG if traineddata.exists() else "eng"


def parse_args():
    parser = argparse.ArgumentParser(
        description="v2 invoice field extraction using YOLO field detection and a custom Tesseract OCR model."
    )
    parser.add_argument("--image_path", "--image", default="images", help="Path to image file or image folder.")
    parser.add_argument("--output", default="invoice_data_v2.xlsx", help="Excel output path. Use empty string to skip.")
    parser.add_argument("--json-output", default=None, help="Optional JSON output path.")
    parser.add_argument("--model", default=existing_default_model(), help="YOLO v2 field detector checkpoint.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--rotate", type=float, default=0.0, help="Rotate image by this angle before detection.")
    parser.add_argument("--crop-dir", default="crops_v2", help="Directory to save detected field crops. Use empty string to skip.")
    parser.add_argument("--no-ocr", action="store_true", help="Only detect and save crops; do not run OCR.")
    parser.add_argument("--tesseract", default=DEFAULT_TESSERACT, help="Tesseract executable path.")
    parser.add_argument("--tessdata-dir", default=DEFAULT_TESSDATA_DIR, help="Folder containing .traineddata files.")
    parser.add_argument("--ocr-lang", default=None, help="Tesseract model name, for example khb_invoice_values.")
    parser.add_argument(
        "--full-field-ocr",
        action="store_true",
        help="Read the entire YOLO field crop instead of the value-focused blue text region.",
    )
    return parser.parse_args()


def normalize_empty_path(value):
    return value if value else None


def tess_config(psm, ocr_lang, tessdata_dir):
    parts = [f"--oem 1 --psm {psm}"]
    if tessdata_dir:
        parts.append(f'--tessdata-dir "{tessdata_dir}"')
    if ocr_lang:
        parts.append(f"-l {ocr_lang}")
    return " ".join(parts)


def read_tesseract(image, configs):
    values = []
    for config in configs:
        text = pytesseract.image_to_string(image, config=config).strip()
        text = " ".join(text.split())
        if text:
            values.append(text)
    return values


def clean_field_text(field_name, candidates):
    if not candidates:
        return ""

    if field_name in {"Invoice No", "Sale Order", "Vender Code"}:
        digits = [re.sub(r"\D", "", value) for value in candidates]
        digits = [value for value in digits if value]
        if not digits:
            return ""
        if field_name == "Sale Order":
            best = max(digits, key=lambda value: (len(value), value))
            return best.zfill(10) if len(best) < 10 else best
        if field_name == "Invoice No":
            best = max(digits, key=lambda value: (len(value), value))
            return best if len(best) >= 6 else ""
        best = max(digits, key=len)
        return best if len(best) >= 4 else ""

    if field_name == "Invoice Date":
        for value in candidates:
            match = re.search(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", value)
            if match:
                day, month, year = match.groups()
                return f"{int(day):02d}.{int(month):02d}.{year}"
        return ""

    if field_name == "Dealer Code":
        cleaned = [re.sub(r"[^A-Za-z0-9]", "", value).upper() for value in candidates]
        cleaned = [
            value for value in cleaned
            if 3 <= len(value) <= 6 and any(char.isalpha() for char in value)
        ]
        return min(cleaned, key=lambda value: abs(len(value) - 4)) if cleaned else ""

    if field_name == "Vehicle Code":
        cleaned = [re.sub(r"[^A-Za-z0-9-]", "", value).upper() for value in candidates]
        cleaned = [
            value for value in cleaned
            if len(value) >= 5 and ("-" in value or any(char.isalpha() for char in value))
        ]
        return max(cleaned, key=len) if cleaned else ""

    return candidates[0]


def extract_text_from_cropped_image_v2(
    cropped_image_path,
    field_name=None,
    tesseract_cmd=None,
    ocr_lang=None,
    tessdata_dir=None,
    value_only=False,
):
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    crop = cv2.imread(str(cropped_image_path))
    if crop is None:
        raise FileNotFoundError(f"Could not load image: {cropped_image_path}")

    if value_only:
        crop = extractor.crop_value_region_for_ocr(crop, field_name)

    configs = [
        tess_config(8, ocr_lang, tessdata_dir),
        tess_config(7, ocr_lang, tessdata_dir),
        tess_config(13, ocr_lang, tessdata_dir),
    ]
    candidates = read_tesseract(crop, configs)
    value = clean_field_text(field_name, candidates)
    if value:
        return value

    processed = extractor.preprocess_for_ocr(crop)
    if processed is not None:
        candidates = read_tesseract(processed, configs)
        value = clean_field_text(field_name, candidates)
        if value:
            return value

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        cv2.imwrite(tmp_file.name, crop)
        tmp_path = tmp_file.name
    try:
        return ORIGINAL_EXTRACTOR_OCR(
            tmp_path,
            field_name,
            tesseract_cmd,
            ocr_lang,
            tessdata_dir,
            False,
        )
    finally:
        os.unlink(tmp_path)


def run_extraction(args):
    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract

    extractor.extract_text_from_cropped_image = extract_text_from_cropped_image_v2

    image_paths = extractor.find_image_paths(args.image_path)
    ocr_lang = args.ocr_lang or existing_default_ocr_lang(args.tessdata_dir)
    crop_dir = normalize_empty_path(args.crop_dir)
    results = []

    print(f"YOLO model: {args.model}")
    if args.no_ocr:
        print("OCR: disabled")
    else:
        print(f"Tesseract language: {ocr_lang}")
        print(f"Tessdata dir: {args.tessdata_dir}")
    print()

    for image_path in image_paths:
        try:
            result = extractor.extract_from_file(
                image_path=image_path,
                model_path=args.model,
                conf=args.conf,
                rotate_angle=args.rotate,
                crop_dir=crop_dir,
                extract_text=not args.no_ocr,
                tesseract_cmd=args.tesseract,
                ocr_lang=ocr_lang,
                tessdata_dir=args.tessdata_dir,
                value_only_ocr=not args.full_field_ocr,
            )
            results.append(result)
            print(f"Processed: {result['filename']}")
            # if not args.no_ocr:
            #     for field_name in extractor.FIELDS:
            #         print(f"  {field_name}: {result['data'].get(field_name, '')}")
            #     if result["missing_fields"]:
            #         print(f"  Missing: {', '.join(result['missing_fields'])}")
            # if crop_dir and result.get("saved_crops"):
            #     print(f"  Crops: {len(result['saved_crops'])} saved to {crop_dir}")
            print()
        except Exception as exc:
            print(f"Error processing {image_path}: {exc}")

    return results


def write_outputs(results, output_path, json_output_path, no_ocr):
    payload = {"count": len(results), "results": results}

    if output_path and results and not no_ocr:
        rows = [{"filename": result["filename"], **result["data"]} for result in results]
        pd.DataFrame(rows).to_excel(output_path, index=False)
        print(f"Excel saved: {output_path}")

    if json_output_path:
        Path(json_output_path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON saved: {json_output_path}")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    results = run_extraction(args)
    write_outputs(results, normalize_empty_path(args.output), args.json_output, args.no_ocr)


if __name__ == "__main__":
    main()
