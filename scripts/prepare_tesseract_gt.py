import argparse
import re
import sys
from pathlib import Path

import cv2
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import FIELDS, convert_crop_for_text


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Tesseract line-image ground truth from YOLO invoice crops."
    )
    parser.add_argument(
        "--crops-dir",
        default="crops_v2",
        help="Directory containing saved YOLO field crops.",
    )
    parser.add_argument(
        "--truth-xlsx",
        default="invoice_data_v2.xlsx",
        help="Excel file with filename and field-value columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="training/tesseract/khb_invoice-ground-truth",
        help="Output directory for .png + .gt.txt training pairs.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include fields whose Excel truth value is empty.",
    )
    return parser.parse_args()


def normalize_field_name(value):
    return value.strip().replace("_", " ")


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def read_truth_rows(path):
    workbook = openpyxl.load_workbook(path, data_only=True)
    sheet = workbook.active
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    headers = [str(value).strip() if value is not None else "" for value in rows[0]]
    filename_index = headers.index("filename") if "filename" in headers else None
    field_indexes = {
        field_name: headers.index(field_name)
        for field_name in FIELDS
        if field_name in headers
    }
    if filename_index is None:
        raise ValueError(f"{path} must contain a filename column.")
    missing = [field_name for field_name in FIELDS if field_name not in field_indexes]
    if missing:
        raise ValueError(f"{path} is missing field columns: {', '.join(missing)}")

    truth = {}
    for row in rows[1:]:
        filename = row[filename_index]
        if not filename:
            continue
        stem = Path(str(filename)).stem
        truth[stem] = {}
        for field_name, column_index in field_indexes.items():
            value = row[column_index] if column_index < len(row) else None
            truth[stem][field_name] = "" if value is None else str(value).strip()
    return truth


def parse_flat_crop(path):
    stem = path.stem
    for field_name in sorted(FIELDS, key=len, reverse=True):
        field_token = field_name.replace(" ", "_")
        marker = f"_{field_token}_"
        if marker in stem:
            image_stem = stem.split(marker, 1)[0]
            return image_stem, field_name
    return None, None


def parse_nested_crop(path):
    if normalize_field_name(path.stem) not in FIELDS:
        return None, None
    folder = path.parent.name
    image_stem = folder.rsplit("_", 1)[0] if "_" in folder else folder
    return image_stem, normalize_field_name(path.stem)


def iter_crops(crops_dir):
    for crop_path in sorted(Path(crops_dir).glob("**/*")):
        if not crop_path.is_file() or crop_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image_stem, field_name = parse_flat_crop(crop_path)
        if not image_stem:
            image_stem, field_name = parse_nested_crop(crop_path)
        if image_stem and field_name:
            yield crop_path, image_stem, field_name


def write_ground_truth(crop_path, output_dir, sample_id, text):
    crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if crop is None:
        raise FileNotFoundError(f"Could not read crop image: {crop_path}")

    line_image = convert_crop_for_text(crop)
    image_path = output_dir / f"{sample_id}.png"
    text_path = output_dir / f"{sample_id}.gt.txt"
    if not cv2.imwrite(str(image_path), line_image):
        raise OSError(f"Could not write line image: {image_path}")
    text_path.write_text(text + "\n", encoding="utf-8")


def main():
    args = parse_args()
    truth = read_truth_rows(Path(args.truth_xlsx))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for crop_path, image_stem, field_name in iter_crops(args.crops_dir):
        text = truth.get(image_stem, {}).get(field_name)
        if text is None:
            skipped += 1
            continue
        if not text and not args.include_empty:
            skipped += 1
            continue

        sample_id = safe_name(f"{image_stem}_{field_name.replace(' ', '_')}")
        write_ground_truth(crop_path, output_dir, sample_id, text)
        written += 1

    print(f"Wrote {written} Tesseract ground-truth pairs to {output_dir}")
    print(f"Skipped {skipped} crops without usable truth text")
    if written < 10:
        print("Warning: this is a very small OCR dataset; add more corrected crops for real accuracy.")


if __name__ == "__main__":
    main()
