# Tesseract Training for Invoice Fields

This folder adds a separate Tesseract fine-tuning workflow. It does not replace
the current YOLO field detector or the existing `main.py`.

## What to Train

Your current pipeline has two parts:

1. YOLO detects the invoice field crop, such as `Invoice No` or `Vehicle Code`.
2. Tesseract reads the text inside that crop.

Train YOLO when field boxes are wrong. Fine-tune Tesseract when boxes are right
but text is misread.

## 1. Create Field Crops

Use the existing app to save crops without changing OCR behavior:

```bash
python main.py --image_path images --crop-dir crops --no-ocr
```

## 2. Prepare Ground Truth Files

```bash
python tesseract_training/prepare_tesseract_gt.py --crops-dir crops
```

This creates:

```text
tesseract_training/ground_truth/
  05_Invoice_No_0.png
  05_Invoice_No_0.gt.txt
  labels_manifest.csv
```

Use `labels_manifest.csv` as a checklist. You can either open each `.gt.txt`
file and type the exact expected text for that crop, or fill the
`text_to_type` column in `labels_manifest.csv` and run:

```bash
python tesseract_training/apply_manifest_labels.py
```

Example `.gt.txt` content:

```text
24000123
```

Keep one text line per crop. Do not include the field name unless it is visible
inside the crop and you want Tesseract to learn it.

## 3. Train Custom Tesseract Data

For best results, make value-focused crops first. This removes much of the red
label/border area and keeps more attention on the blue value text:

```bash
python tesseract_training/make_value_crops.py
```

For English letters and numbers:

```bash
python tesseract_training/train_tesseract_invoice.py \
  --ground-truth-dir tesseract_training/value_ground_truth \
  --base-lang eng \
  --lang-name khb_invoice_values \
  --tessdata-dir tesseract_training/tessdata_best \
  --max-iterations 1000
```

For Khmer-heavy fields, start from Khmer instead:

```bash
python tesseract_training/train_tesseract_invoice.py \
  --base-lang khm \
  --lang-name khb_invoice_khm \
  --max-iterations 1000
```

The output is written to:

```text
tesseract_training/output/khb_invoice_values.traineddata
```

## 4. Use the Custom Model

Copy the traineddata file into a tessdata folder, for example:

```bash
cp tesseract_training/output/khb_invoice_values.traineddata /opt/homebrew/share/tessdata/
```

Then OCR can use:

```text
-l khb_invoice_values
```

Your current `main.py` hardcodes field-specific Tesseract configs. To keep that
previous version intact, test the traineddata first with:

```bash
tesseract tesseract_training/value_ground_truth/05_Invoice_No_0.png stdout \
  --psm 7 --tessdata-dir tesseract_training/output -l khb_invoice_values
```

After the trained model reads better than `eng`, wire `-l khb_invoice` into the
field configs in a new app copy or a small OCR helper.

## Data Tips

- Use at least 50 to 100 crop samples for each common field style.
- Label the exact text, including leading zeros.
- Keep separate training runs for very different scripts if needed: English code
  fields and Khmer text fields may work better as separate traineddata files.
- If recognition is bad only for date/code formats, first try better Tesseract
  config and preprocessing before long training.
