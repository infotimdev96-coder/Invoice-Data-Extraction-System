# Invoice Field Detection System

This project uses a custom YOLO model to detect invoice fields and save each
detected field as a cropped image. It does not perform text recognition.

## Features

- Train YOLOv8 on annotated invoice images.
- Detect six invoice fields:
  - Invoice No
  - Invoice Date
  - Dealer Code
  - Sale Order
  - Vender Code
  - Vehicle Code
- Save detected field crops for later use.
- Print detection confidence scores, bounding boxes, and missing fields as JSON.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Detect Fields

```bash
python main.py \
  --image_path images \
  --model runs/detect/khb_field_model_6fields/weights/best.pt \
  --conf 0.25 \
  --crop-dir crops
```

Use `--crop-dir ""` to print detections without saving crop images.

## Train Tesseract OCR For Field Text

YOLO training detects the invoice fields. Tesseract training needs separate
ground truth: each saved field crop must have the exact text value from that
crop.

This project now uses the official Tesseract 5 `tesstrain` workflow:

```bash
brew install make wget
git clone https://github.com/tesseract-ocr/tesstrain.git tools/tesstrain
mkdir -p training/tessdata_best
wget -O training/tessdata_best/eng.traineddata \
  https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata
```

Create Tesseract ground-truth pairs from the existing YOLO crops and Excel truth
file:

```bash
source venv/bin/activate
python scripts/prepare_tesseract_gt.py \
  --crops-dir crops_v2 \
  --truth-xlsx invoice_data_v2.xlsx \
  --output-dir training/tesseract/khb_invoice-ground-truth
```

Start fine-tuning from the trainable English `tessdata_best` model:

```bash
MODEL_NAME=khb_invoice_best \
GROUND_TRUTH_DIR=training/tesseract/khb_invoice-ground-truth \
TESSDATA=training/tessdata_best \
MAX_ITERATIONS=300 \
scripts/train_tesseract_khb.sh
```

The main trained model is written to:

```text
tools/tesstrain/data/khb_invoice_best.traineddata
```

Smoke-test the trained model on one prepared crop:

```bash
tesseract training/tesseract/khb_invoice-ground-truth/05_Invoice_Date.png stdout \
  --tessdata-dir tools/tesstrain/data \
  -l khb_invoice_best \
  --psm 7
```

The current starter dataset has only a small number of non-empty text labels.
Use it to verify the pipeline, then add many more corrected crop/text pairs
before relying on OCR accuracy.

## Train The YOLO Model

```bash
python train_khb_model.py \
  --data khb_dataset/data.yaml \
  --epochs 100 \
  --imgsz 960 \
  --batch 1 \
  --name khb_field_model_6fields
```

The dataset labels are used only during training. Production field detection
uses the trained YOLO checkpoint and invoice images.
