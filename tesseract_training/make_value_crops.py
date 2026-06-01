import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create value-focused crops from YOLO invoice field crops."
    )
    parser.add_argument(
        "--input-dir",
        default="tesseract_training/ground_truth",
        help="Directory with original field crops and .gt.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        default="tesseract_training/value_ground_truth",
        help="Directory where value-focused crops and labels are written.",
    )
    return parser.parse_args()


def crop_value_region(image_path):
    image = Image.open(image_path).convert("RGB")
    arr = np.array(image)
    height, width = arr.shape[:2]
    red, green, blue = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    blue_text_mask = (blue > red + 10) & (blue > green + 8) & (blue > 70)
    name = image_path.name

    if "Invoice_Date" in name or "Invoice_No" in name:
        cutoff = int(height * 0.10)
    elif "Vehicle_Code" in name or "Vender_Code" in name:
        cutoff = int(height * 0.22)
    else:
        cutoff = int(height * 0.42)

    blue_text_mask[:cutoff, :] = False
    ys, xs = np.where(blue_text_mask)

    if len(xs) < 10:
        return image.crop((0, cutoff, width, height))

    x_min = max(int(xs.min()) - 10, 0)
    x_max = min(int(xs.max()) + 11, width)
    y_min = max(int(ys.min()) - 7, 0)
    y_max = min(int(ys.max()) + 8, height)
    return image.crop((x_min, y_min, x_max, y_max))


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for image_path in sorted(input_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        crop = crop_value_region(image_path)
        crop.save(output_dir / image_path.name)

        gt_path = image_path.with_suffix(".gt.txt")
        if gt_path.exists():
            (output_dir / f"{image_path.stem}.gt.txt").write_text(
                gt_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

        count += 1

    print(f"Created {count} value-focused crops in: {output_dir}")


if __name__ == "__main__":
    main()
