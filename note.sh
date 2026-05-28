#!/bin/zsh

# KHB invoice/delivery-order OCR + YOLO training notes
# Run commands from this project folder:
# /Users/timdev/Desktop/works/Python Projects/Invoice-Data-Extraction-System


# 1) Activate environment
source venv/bin/activate


# 2) Install missing tools if needed
pip install -r requirements.txt
pip install labelImg


# 3) Dataset folders
# Put images here:
#   khb_dataset/train/images/
#   khb_dataset/valid/images/
#   khb_dataset/test/images/
#
# Put YOLO label .txt files here:
#   khb_dataset/train/labels/
#   khb_dataset/valid/labels/
#   khb_dataset/test/labels/
#
# Image and label names must match:
#   khb_dataset/train/images/invoice1.jpg
#   khb_dataset/train/labels/invoice1.txt


# 4) Label classes
# Use these 8 classes exactly:
#   0 Invoice No
#   1 Invoice Date
#   2 Dealer Code
#   3 Sale Order
#   4 Vender Code
#   5 Vehicle Code
#   6 Route
#   7 Warehouse
#
# Draw boxes around the value text only, not the title.
# Example:
#   Invoice No    -> box around 70239990
#   Invoice Date  -> box around 27.02.2026
#   Dealer Code   -> box around BVL6
#   Sale Order    -> box around 0030068003
#   Vender Code   -> box around 100569
#   Vehicle Code  -> box around 3A-4336
#   Route         -> box around KHB-Brewery->R4-Bavel
#   Warehouse     -> box around WF11-FG Warehouse


# 5) Open LabelImg to create labels
labelImg
#
# Inside LabelImg:
# - Open Dir: khb_dataset/train/images
# - Change Save Dir: khb_dataset/train/labels
# - Save format: YOLO
# - Draw one box per field value
# - Save each image
#
# Repeat for:
# - khb_dataset/valid/images -> khb_dataset/valid/labels
# - khb_dataset/test/images  -> khb_dataset/test/labels


# 6) Recommended data split
# Use about:
# - 80% images in train/images
# - 10% images in valid/images
# - 10% images in test/images


# 7) Train your YOLO model
python train_khb_model.py --data khb_dataset/data.yaml --epochs 100 --imgsz 960


# 8) After training, your best model will be around:
# runs/detect/khb_field_model/weights/best.pt


# 9) Use your trained YOLO model to detect fields, then OCR the detected crops
python main.py \
  --image "/Users/timdev/Downloads/new-inv-image.jpeg" \
  --model runs/detect/khb_field_model/weights/best.pt \
  --conf 0.25 \
  --output khb_model_output.xlsx \
  --crops-dir khb_model_crops \
  --full-text-output khb_model_full_text.txt \
  --debug-image khb_model_debug.jpg


# 10) Run current fixed-region OCR extractor on one image
python main.py \
  --image "/Users/timdev/Downloads/new-inv-image.jpeg" \
  --output khb_invoice_data.xlsx \
  --full-text-output khb_full_text.txt \
  --debug-image khb_debug_regions.jpg


# 11) If a scan is slightly rotated, try:
python main.py \
  --image "/Users/timdev/Downloads/new-inv-image.jpeg" \
  --rotate -1 \
  --output khb_invoice_data_rotated.xlsx \
  --full-text-output khb_full_text_rotated.txt \
  --debug-image khb_debug_regions_rotated.jpg


python train_khb_model.py --data khb_dataset/data.yaml --epochs 1 --imgsz 640 --batch 1 --name khb_test_run
# For real training, run:
ython train_khb_model.py --data khb_dataset/data.yaml --epochs 100 --imgsz 960 --batch 1



python main.py \
  --image "/Users/timdev/Desktop/works/Python Projects/Invoice-Data-Extraction-System/khb_dataset/train/images/01.jpg" \
  --output khb_invoice_data.xlsx

python main.py \
  --image "/Users/timdev/Desktop/works/Python Projects/Invoice-Data-Extraction-System/khb_dataset/train/images/02.jpg" \
  --output khb_invoice_data.xlsx