#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-khb_invoice_best}"
START_MODEL="${START_MODEL:-eng}"
GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR:-training/tesseract/khb_invoice-ground-truth}"
TESSDATA="${TESSDATA:-training/tessdata_best}"
TESSTRAIN_DIR="${TESSTRAIN_DIR:-tools/tesstrain}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2000}"
RATIO_TRAIN="${RATIO_TRAIN:-0.85}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GROUND_TRUTH_DIR="$(cd "$PROJECT_DIR" && mkdir -p "$(dirname "$GROUND_TRUTH_DIR")" && python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$GROUND_TRUTH_DIR")"
TESSDATA="$(cd "$PROJECT_DIR" && python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$TESSDATA")"
TESSTRAIN_DIR="$(cd "$PROJECT_DIR" && python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$TESSTRAIN_DIR")"

SAFE_PATH_DIR="/private/tmp/khb_invoice_tesseract"
mkdir -p "$SAFE_PATH_DIR"
ln -sfn "$GROUND_TRUTH_DIR" "$SAFE_PATH_DIR/ground_truth"
ln -sfn "$TESSDATA" "$SAFE_PATH_DIR/tessdata"
GROUND_TRUTH_DIR="$SAFE_PATH_DIR/ground_truth"
TESSDATA="$SAFE_PATH_DIR/tessdata"

if [ ! -d "$TESSTRAIN_DIR" ]; then
  echo "Missing $TESSTRAIN_DIR"
  echo "Clone it with: git clone https://github.com/tesseract-ocr/tesstrain.git $TESSTRAIN_DIR"
  exit 1
fi

if [ ! -d "$GROUND_TRUTH_DIR" ]; then
  echo "Missing ground truth: $GROUND_TRUTH_DIR"
  echo "Create it with: source venv/bin/activate && python scripts/prepare_tesseract_gt.py"
  exit 1
fi

if ! command -v gmake >/dev/null 2>&1; then
  echo "GNU make is required. Install it with: brew install make"
  exit 1
fi

if [ ! -f "$TESSDATA/${START_MODEL}.traineddata" ]; then
  echo "Missing start model: $TESSDATA/${START_MODEL}.traineddata"
  exit 1
fi

gmake -C "$TESSTRAIN_DIR" training \
  MODEL_NAME="$MODEL_NAME" \
  START_MODEL="$START_MODEL" \
  TESSDATA="$TESSDATA" \
  GROUND_TRUTH_DIR="$GROUND_TRUTH_DIR" \
  MAX_ITERATIONS="$MAX_ITERATIONS" \
  RATIO_TRAIN="$RATIO_TRAIN"

gmake -C "$TESSTRAIN_DIR" traineddata \
  MODEL_NAME="$MODEL_NAME" \
  START_MODEL="$START_MODEL" \
  TESSDATA="$TESSDATA" \
  GROUND_TRUTH_DIR="$GROUND_TRUTH_DIR"

echo "Best model output:"
find "$TESSTRAIN_DIR/data" -path "*${MODEL_NAME}*.traineddata" -print
