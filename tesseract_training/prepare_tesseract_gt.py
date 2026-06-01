import argparse
import csv
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Tesseract LSTM ground-truth files from invoice field crops."
    )
    parser.add_argument(
        "--crops-dir",
        default="crops",
        help="Directory containing cropped invoice field images.",
    )
    parser.add_argument(
        "--output-dir",
        default="tesseract_training/ground_truth",
        help="Directory where Tesseract training images and .gt.txt files are written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files in the output directory.",
    )
    return parser.parse_args()


def safe_stem(path):
    return path.stem.replace(" ", "_")


def prepare_ground_truth(crops_dir, output_dir, overwrite=False):
    crops_dir = Path(crops_dir)
    output_dir = Path(output_dir)

    if not crops_dir.exists():
        raise FileNotFoundError(f"Missing crops directory: {crops_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path for path in crops_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise FileNotFoundError(f"No crop images found in: {crops_dir}")

    manifest_path = output_dir / "labels_manifest.csv"
    rows = []

    for image_path in images:
        target_image = output_dir / image_path.name
        target_gt = output_dir / f"{safe_stem(image_path)}.gt.txt"

        if overwrite or not target_image.exists():
            shutil.copy2(image_path, target_image)

        if overwrite or not target_gt.exists():
            target_gt.write_text("", encoding="utf-8")

        rows.append(
            {
                "image": target_image.name,
                "ground_truth_file": target_gt.name,
                "text_to_type": "",
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image", "ground_truth_file", "text_to_type"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), manifest_path


def main():
    args = parse_args()
    count, manifest_path = prepare_ground_truth(
        args.crops_dir,
        args.output_dir,
        args.overwrite,
    )
    print(f"Prepared {count} crop images.")
    print(f"Ground-truth folder: {Path(args.output_dir).resolve()}")
    print(f"Use this checklist while labeling: {manifest_path}")
    print("Fill the expected text in each matching .gt.txt file before training.")


if __name__ == "__main__":
    main()
