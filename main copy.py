import argparse
import re
import shutil
from pathlib import Path

import cv2
import pandas as pd
import pytesseract


COLUMNS = [
    "Invoice No",
    "Invoice Date",
    "Dealer Code",
    "Sale Order",
    "Vender Code",
    "Vehicle Code",
]

# Normalized crop boxes for the KHB delivery-order layout:
# (left, top, right, bottom), each value from 0.0 to 1.0.
FIELD_REGIONS = {
    "Invoice No": (0.675, 0.140, 0.790, 0.160),
    "Invoice Date": (0.855, 0.140, 0.965, 0.160),
    "Dealer Code": (0.515, 0.207, 0.630, 0.228),
    "Sale Order": (0.515, 0.257, 0.670, 0.278),
    "Vender Code": (0.515, 0.315, 0.625, 0.345),
    "Vehicle Code": (0.745, 0.315, 0.850, 0.345),
}


def crop_region(image, region):
    height, width = image.shape[:2]
    left, top, right, bottom = region
    return image[
        int(top * height) : int(bottom * height),
        int(left * width) : int(right * width),
    ]


def upscale_crop(crop, scale=10):
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)


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


def clean_dealer_code(value):
    value = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    value = value.replace("S", "6") if value.endswith("S") else value
    if value.startswith("BYV") and len(value) >= 5:
        value = "B" + value[2:]
    return value


def clean_sale_order(value):
    digits = only_digits(value)
    if len(digits) < 10:
        digits = digits.zfill(10)
    return digits


def clean_vender_code(value):
    digits = only_digits(value)
    if len(digits) == 5 and digits.startswith("10"):
        digits = digits[:2] + "0" + digits[2:]
    return digits


def clean_vehicle_code(value):
    value = re.sub(r"[^A-Za-z0-9-]", "", value).upper()
    if "-" not in value and len(value) >= 6:
        value = f"{value[:2]}-{value[2:]}"
    return value


def extract_field(crop, field_name):
    if field_name == "Invoice No":
        image = preprocess(crop, channel="gray", scale=10, use_clahe=True)
        return clean_invoice_no(ocr_text(image, "0123456789", psm=8))

    if field_name == "Invoice Date":
        image = preprocess(crop, channel="gray", scale=10, use_clahe=True)
        return clean_invoice_date(ocr_text(image, "0123456789./-", psm=8))

    if field_name == "Dealer Code":
        image = preprocess(crop, channel="gray", scale=10, use_clahe=False)
        return clean_dealer_code(ocr_text(image, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", psm=8))

    if field_name == "Sale Order":
        image = preprocess(crop, channel="gray", scale=15, use_clahe=True)
        return clean_sale_order(ocr_text(image, "0123456789", psm=8))

    if field_name == "Vender Code":
        image = preprocess(crop, channel="gray", scale=10, use_clahe=True)
        return clean_vender_code(ocr_text(image, "0123456789", psm=7))

    if field_name == "Vehicle Code":
        image = preprocess(crop, channel="red", scale=15, use_clahe=True)
        return clean_vehicle_code(ocr_text(image, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-", psm=7))

    raise ValueError(f"Unsupported field: {field_name}")


def save_crops(image, crops_dir):
    crops_dir.mkdir(parents=True, exist_ok=True)
    for crop_file in crops_dir.glob("*.png"):
        crop_file.unlink()

    crops = {}
    for field_name, region in FIELD_REGIONS.items():
        crop = crop_region(image, region)
        crops[field_name] = crop
        crop_path = crops_dir / f"{field_name.replace(' ', '_')}.png"
        cv2.imwrite(str(crop_path), crop)
    return crops


def write_excel(data, output_path):
    df = pd.DataFrame([data], columns=COLUMNS)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
        worksheet = writer.sheets["Sheet1"]
        for column_cells in worksheet.columns:
            max_length = max(len(str(cell.value or "")) for cell in column_cells)
            worksheet.column_dimensions[column_cells[0].column_letter].width = max(
                max_length + 2, 14
            )


def parse_args():
    default_tesseract = shutil.which("tesseract")

    parser = argparse.ArgumentParser(
        description="Extract selected KHB delivery-order fields from an image."
    )
    parser.add_argument(
        "--image",
        default="/Users/timdev/Downloads/new-inv-image.jpeg",
        help="Path to the delivery-order image.",
    )
    parser.add_argument(
        "--output",
        default="invoice_data.xlsx",
        help="Path to the Excel file to create.",
    )
    parser.add_argument(
        "--crops-dir",
        default="savedimages",
        help="Directory where field crop images are saved.",
    )
    parser.add_argument(
        "--tesseract",
        default=default_tesseract,
        help="Path to the tesseract executable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    crops = save_crops(image, Path(args.crops_dir))
    data = {field_name: extract_field(crops[field_name], field_name) for field_name in COLUMNS}

    write_excel(data, Path(args.output))

    print(f"Saved crop images to: {args.crops_dir}")
    print(f"Saved OCR result to: {args.output}")
    print("Extracted data:")
    for field_name in COLUMNS:
        print(f"- {field_name}: {data[field_name]}")


if __name__ == "__main__":
    main()
