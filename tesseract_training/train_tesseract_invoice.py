import argparse
import shutil
import subprocess
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Tesseract LSTM OCR on labeled invoice field crops."
    )
    parser.add_argument(
        "--ground-truth-dir",
        default="tesseract_training/ground_truth",
        help="Directory containing crop images and matching .gt.txt files.",
    )
    parser.add_argument(
        "--lang-name",
        default="khb_invoice",
        help="Name for the exported traineddata file.",
    )
    parser.add_argument(
        "--base-lang",
        default="eng",
        help="Installed Tesseract language to fine-tune from, usually eng or khm.",
    )
    parser.add_argument(
        "--tessdata-dir",
        default="/opt/homebrew/share/tessdata",
        help="Directory containing base .traineddata files.",
    )
    parser.add_argument("--psm", default="7", help="Page segmentation mode for field crops.")
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        default="tesseract_training/output",
        help="Directory where lstmf/checkpoints/final traineddata are written.",
    )
    return parser.parse_args()


def run(command):
    print(" ".join(str(part) for part in command))
    subprocess.run(command, check=True)


def require_tool(name):
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Missing required Tesseract training tool: {name}")
    return path


def collect_training_pairs(ground_truth_dir):
    images = sorted(
        path for path in ground_truth_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    pairs = []

    for image_path in images:
        gt_path = image_path.with_suffix(".gt.txt")
        if not gt_path.exists():
            print(f"Skipping {image_path.name}: missing {gt_path.name}")
            continue

        text = gt_path.read_text(encoding="utf-8").strip()
        if not text:
            print(f"Skipping {image_path.name}: empty ground-truth text")
            continue

        pairs.append((image_path, gt_path))

    if not pairs:
        raise RuntimeError(
            "No usable training pairs found. Each crop needs a matching non-empty .gt.txt file."
        )

    return pairs


def write_box_file(image_path, gt_path):
    text = gt_path.read_text(encoding="utf-8").strip()
    with Image.open(image_path) as image:
        width, height = image.size
    chars = [char for char in text if char != "\n"]
    if not chars:
        raise RuntimeError(f"Cannot create box file for empty text: {gt_path}")

    step = max(width // len(chars), 1)
    lines = []
    for index, char in enumerate(chars):
        x_min = index * step
        x_max = width if index == len(chars) - 1 else min((index + 1) * step, width)
        lines.append(f"{char} {x_min} 0 {x_max} {height} 0")

    box_path = image_path.with_suffix(".box")
    box_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return box_path


def main():
    args = parse_args()
    require_tool("tesseract")
    require_tool("combine_tessdata")
    require_tool("lstmtraining")

    ground_truth_dir = Path(args.ground_truth_dir)
    output_dir = Path(args.output_dir)
    lstmf_dir = output_dir / "lstmf"
    checkpoint_prefix = output_dir / args.lang_name
    final_traineddata = output_dir / f"{args.lang_name}.traineddata"
    base_traineddata = Path(args.tessdata_dir) / f"{args.base_lang}.traineddata"
    base_lstm = output_dir / f"{args.base_lang}.lstm"
    train_list = output_dir / "train_list.txt"

    if not ground_truth_dir.exists():
        raise FileNotFoundError(f"Missing ground-truth directory: {ground_truth_dir}")
    if not base_traineddata.exists():
        raise FileNotFoundError(f"Missing base traineddata: {base_traineddata}")

    output_dir.mkdir(parents=True, exist_ok=True)
    lstmf_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_training_pairs(ground_truth_dir)

    run(["combine_tessdata", "-e", str(base_traineddata), str(base_lstm)])

    lstmf_paths = []
    for image_path, gt_path in pairs:
        write_box_file(image_path, gt_path)
        output_base = lstmf_dir / image_path.stem
        shutil.copy2(gt_path, output_base.with_suffix(".gt.txt"))
        run([
            "tesseract",
            str(image_path),
            str(output_base),
            "--psm",
            str(args.psm),
            "lstm.train",
        ])
        lstmf_path = output_base.with_suffix(".lstmf")
        if lstmf_path.exists():
            lstmf_paths.append(lstmf_path)

    train_list.write_text(
        "\n".join(str(path.resolve()) for path in lstmf_paths) + "\n",
        encoding="utf-8",
    )

    run([
        "lstmtraining",
        "--model_output",
        str(checkpoint_prefix),
        "--continue_from",
        str(base_lstm),
        "--traineddata",
        str(base_traineddata),
        "--train_listfile",
        str(train_list),
        "--max_iterations",
        str(args.max_iterations),
    ])

    checkpoint_path = Path(f"{checkpoint_prefix}_checkpoint")
    run([
        "lstmtraining",
        "--stop_training",
        "--continue_from",
        str(checkpoint_path),
        "--traineddata",
        str(base_traineddata),
        "--model_output",
        str(final_traineddata),
    ])

    print(f"Created: {final_traineddata.resolve()}")
    print("Use it by copying the file into your tessdata directory or by setting TESSDATA_PREFIX.")


if __name__ == "__main__":
    main()
