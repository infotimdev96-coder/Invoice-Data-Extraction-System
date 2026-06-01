import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy labels_manifest.csv text_to_type values into Tesseract .gt.txt files."
    )
    parser.add_argument(
        "--manifest",
        default="tesseract_training/ground_truth/labels_manifest.csv",
        help="Path to labels_manifest.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    updated = 0
    skipped = 0
    base_dir = manifest_path.parent

    with manifest_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            gt_name = row.get("ground_truth_file", "").strip()
            text = row.get("text_to_type", "")

            if not gt_name or not text.strip():
                skipped += 1
                continue

            gt_path = base_dir / gt_name
            gt_path.write_text(text.strip() + "\n", encoding="utf-8")
            updated += 1

    print(f"Updated {updated} ground-truth files.")
    if skipped:
        print(f"Skipped {skipped} rows with missing file names or empty labels.")


if __name__ == "__main__":
    main()
