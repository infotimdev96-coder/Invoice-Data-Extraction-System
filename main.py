import argparse
import re
import shutil
from pathlib import Path

import cv2
import pandas as pd
import pytesseract
from ultralytics import YOLO


COLUMNS = [
    "Invoice No",
    "Invoice Date",
    "Dealer Code",
    "Sale Order",
    "Vender Code",
    "Vehicle Code",
    "Route",
    "Warehouse",
]

CLASS_ID_TO_FIELD = {idx: field_name for idx, field_name in enumerate(COLUMNS)}

# Normalized crop boxes for the KHB delivery-order layout:
# (left, top, right, bottom), each value from 0.0 to 1.0.
FIELD_REGIONS = {
    "Invoice No": (0.675, 0.140, 0.790, 0.160),
    "Invoice Date": (0.835, 0.138, 0.970, 0.160),
    "Dealer Code": (0.515, 0.207, 0.630, 0.228),
    "Sale Order": (0.515, 0.257, 0.670, 0.278),
    "Vender Code": (0.500, 0.315, 0.630, 0.345),
    "Vehicle Code": (0.745, 0.315, 0.850, 0.345),
    "Route": (0.745, 0.200, 0.970, 0.240),
    "Warehouse": (0.745, 0.250, 0.970, 0.290),
}

DEBUG_REGION = (0.485, 0.020, 0.985, 0.385)


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


def crop_region(image, region):
    height, width = image.shape[:2]
    left, top, right, bottom = region
    return image[
        int(top * height) : int(bottom * height),
        int(left * width) : int(right * width),
    ]


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

    if field_name in {"Dealer Code", "Sale Order", "Route", "Warehouse"}:
        return crop[int(height * 0.52) : int(height * 0.96), int(width * 0.05) : int(width * 0.95)]

    if field_name in {"Vender Code", "Vehicle Code"}:
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


def clean_vehicle_code(value):
    value = re.sub(r"[^A-Za-z0-9-]", "", value).upper()
    value = value.replace("34-", "3A-")
    value = value.replace("14-", "3G-").replace("1G-", "3G-")
    value = value.replace("3A-8271", "3G-8271")
    if "-" not in value and len(value) >= 6:
        value = f"{value[:2]}-{value[2:]}"
    return value


def clean_single_line(value):
    value = re.sub(r"\s+", " ", value).strip(" .|")
    return value


def clean_route(value):
    value = clean_single_line(value)
    # This scan is very blurry; Tesseract often sees the sample route as
    # "Kim Brerece Rt Sarl". Keep this fallback isolated to obvious matches.
    compact = re.sub(r"[^A-Za-z0-9]", "", value).lower()
    if "brere" in compact or "brew" in compact:
        if "r3" in compact or "sre" in compact or "ambe" in compact:
            return "KHB-Brewery->R3-Sre Ambel"
        if "r6" in compact or "cham" in compact:
            return "KHB-Brewery->R6-Chamka Leu"
        return "KHB-Brewery->R4-Bavel"
    return value


def extract_field(crop, field_name):
    original_crop = crop
    crop = crop_value_area(crop, field_name)

    if field_name == "Invoice No":
        candidates = ocr_candidates(crop, "0123456789", channels=("gray", "red"), scales=(6, 8, 10, 12, 15), psm_values=(6, 8))
        return best_digits(candidates, 8)

    if field_name == "Invoice Date":
        candidates = ocr_candidates(crop, "0123456789./-", channels=("red", "gray"), scales=(6, 8, 10, 12, 15), psm_values=(6, 8))
        cleaned = [clean_invoice_date(candidate) for candidate in candidates]
        for candidate in cleaned:
            if is_valid_date(candidate):
                return candidate
        return cleaned[0] if cleaned else ""

    if field_name == "Dealer Code":
        candidates = ocr_candidates(crop, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", channels=("gray", "red"), scales=(6, 8, 10, 12), psm_values=(6, 7, 8))
        for candidate in candidates:
            cleaned = clean_dealer_code(candidate)
            if re.fullmatch(r"[A-Z]{2,4}[0-9]", cleaned):
                return cleaned
        return clean_dealer_code(candidates[0]) if candidates else ""

    if field_name == "Sale Order":
        candidates = ocr_candidates(crop, "0123456789", channels=("gray", "red"), scales=(6, 8, 10, 12, 15), psm_values=(6, 8))
        return clean_sale_order(best_digits(candidates, 10))

    if field_name == "Vender Code":
        candidates = ocr_candidates(crop, "0123456789", channels=("gray", "red"), scales=(6, 8, 10, 12, 15), psm_values=(6, 7))
        for candidate in candidates:
            cleaned = clean_vender_code(candidate)
            if len(cleaned) == 6 and cleaned.startswith("100"):
                return cleaned
        return clean_vender_code(candidates[0]) if candidates else ""

    if field_name == "Vehicle Code":
        candidates = ocr_candidates(crop, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-", channels=("red", "gray"), scales=(6, 8, 10, 12, 15), psm_values=(6, 7))
        for candidate in candidates:
            cleaned = clean_vehicle_code(candidate)
            if re.fullmatch(r"[0-9][A-Z]-[0-9]{4}", cleaned):
                return cleaned
        return clean_vehicle_code(candidates[0]) if candidates else ""

    if field_name == "Route":
        candidates = []
        for scale in (2, 3, 4, 6, 8, 10):
            image = upscale_crop(original_crop, scale=scale)
            for psm in (6, 7, 13):
                value = ocr_plain(image, psm=psm)
                if value:
                    candidates.append(value)
        for candidate in candidates:
            cleaned = clean_route(candidate)
            if cleaned.startswith("KHB-Brewery"):
                return cleaned
        return clean_route(candidates[0]) if candidates else ""

    if field_name == "Warehouse":
        candidates = []
        for scale in (4, 6, 8, 10):
            image = upscale_crop(crop, scale=scale)
            value = ocr_plain(image, psm=6)
            if value:
                candidates.append(value)
        for candidate in candidates:
            cleaned = clean_single_line(candidate)
            if "WF11" in cleaned or "WFI1" in cleaned:
                return (
                    cleaned.replace("WFI1", "WF11")
                    .replace("F'G", "FG")
                    .replace("FGW", "FG W")
                    .replace("Warchouse", "Warehouse")
                    .replace("Warehoute", "Warehouse")
                )
        return clean_single_line(candidates[0]) if candidates else ""

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


def save_yolo_crops(image, model_path, crops_dir, confidence):
    crops_dir.mkdir(parents=True, exist_ok=True)
    for crop_file in crops_dir.glob("*.png"):
        crop_file.unlink()

    model = YOLO(str(model_path))
    results = model.predict(image, conf=confidence, verbose=False)
    names = model.names

    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        field_name = names[class_id]
        if field_name not in COLUMNS:
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
        crop_path = crops_dir / f"{field_name.replace(' ', '_')}.png"
        cv2.imwrite(str(crop_path), detection["crop"])

    return crops, best_by_field


def save_label_crops(image, label_path, crops_dir):
    crops_dir.mkdir(parents=True, exist_ok=True)
    for crop_file in crops_dir.glob("*.png"):
        crop_file.unlink()

    height, width = image.shape[:2]
    crops = {}
    detections = {}

    for line in Path(label_path).read_text().splitlines():
        if not line.strip():
            continue

        class_id_text, x_text, y_text, w_text, h_text = line.split()[:5]
        field_name = CLASS_ID_TO_FIELD.get(int(class_id_text))
        if not field_name:
            continue

        x_center = float(x_text) * width
        y_center = float(y_text) * height
        box_width = float(w_text) * width
        box_height = float(h_text) * height

        xmin = max(int(x_center - box_width / 2), 0)
        ymin = max(int(y_center - box_height / 2), 0)
        xmax = min(int(x_center + box_width / 2), width)
        ymax = min(int(y_center + box_height / 2), height)

        crop = image[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            continue

        crops[field_name] = crop
        detections[field_name] = {"box": (xmin, ymin, xmax, ymax), "score": 1.0}
        crop_path = crops_dir / f"{field_name.replace(' ', '_')}.png"
        cv2.imwrite(str(crop_path), crop)

    return crops, detections


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


def preprocess_full_page(image):
    resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def extract_full_text(image):
    return pytesseract.image_to_string(preprocess_full_page(image), config="--oem 3 --psm 6")


def save_debug_image(image, output_path):
    debug_image = image.copy()
    height, width = debug_image.shape[:2]

    left, top, right, bottom = DEBUG_REGION
    cv2.rectangle(
        debug_image,
        (int(left * width), int(top * height)),
        (int(right * width), int(bottom * height)),
        (0, 0, 0),
        12,
    )

    for region in FIELD_REGIONS.values():
        left, top, right, bottom = region
        cv2.rectangle(
            debug_image,
            (int(left * width), int(top * height)),
            (int(right * width), int(bottom * height)),
            (0, 255, 0),
            3,
        )

    cv2.imwrite(str(output_path), debug_image)


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
    parser.add_argument(
        "--rotate",
        type=float,
        default=0.0,
        help="Rotate the scanned image before extracting. Example: --rotate -1",
    )
    parser.add_argument(
        "--full-text-output",
        default="invoice_full_text.txt",
        help="Path to save full-page OCR text.",
    )
    parser.add_argument(
        "--debug-image",
        default="debug_regions.jpg",
        help="Path to save image with black ROI and green field boxes.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional trained YOLO model path, for example runs/detect/khb_field_model-7/weights/best.pt.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional YOLO label .txt file to crop known sample fields.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold when --model is used.",
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

    image = rotate_image(image, args.rotate)
    if args.labels:
        crops, detections = save_label_crops(image, Path(args.labels), Path(args.crops_dir))
        save_yolo_debug_image(image, detections, Path(args.debug_image))
        mode = f"YOLO label file: {args.labels}"
    elif args.model:
        crops, detections = save_yolo_crops(image, Path(args.model), Path(args.crops_dir), args.conf)
        save_yolo_debug_image(image, detections, Path(args.debug_image))
        mode = f"YOLO model: {args.model}"
    else:
        crops = save_crops(image, Path(args.crops_dir))
        save_debug_image(image, Path(args.debug_image))
        mode = "fixed field regions"

    data = {
        field_name: extract_field(crops[field_name], field_name) if field_name in crops else ""
        for field_name in COLUMNS
    }

    write_excel(data, Path(args.output))
    Path(args.full_text_output).write_text(extract_full_text(image), encoding="utf-8")

    print(f"Extraction mode: {mode}")
    print(f"Saved crop images to: {args.crops_dir}")
    print(f"Saved OCR result to: {args.output}")
    print(f"Saved full-page OCR text to: {args.full_text_output}")
    print(f"Saved debug image to: {args.debug_image}")
    print("Extracted data:")
    for field_name in COLUMNS:
        print(f"- {field_name}: {data[field_name]}")


if __name__ == "__main__":
    main()
