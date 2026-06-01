#!/bin/zsh

# KHB delivery-order field detection workflow
# Run commands from:
# /Users/timdev/Desktop/works/Python Projects/Invoice-Data-Extraction-System

source venv/bin/activate


# 1) Dataset rule
# Labels are ONLY for training. Do not use labels in production extraction.
#
# Dataset:
#   khb_dataset/train/images/*.jpg|jpeg|png
#   khb_dataset/train/labels/*.txt
#   khb_dataset/valid/images/*.jpg|jpeg|png
#   khb_dataset/valid/labels/*.txt
#   khb_dataset/test/images/*.jpg|jpeg|png
#   khb_dataset/test/labels/*.txt
#
# File names must match:
#   train/images/02.jpg
#   train/labels/02.txt


# 2) Label classes
# 0 Invoice No
# 1 Invoice Date
# 2 Dealer Code
# 3 Sale Order
# 4 Vender Code
# 5 Vehicle Code
#
# Best practice for this project:
# Draw one box around the whole field cell, like your black boxes.
# The OCR code will crop inside the detected cell to read only the blue value.


# 3) Check dataset quickly
find khb_dataset/train/images -maxdepth 1 -type f | sort
find khb_dataset/train/labels -maxdepth 1 -type f | sort


# 4) Train all labeled template images
# This script clears stale labels.cache automatically.
python train_khb_model.py \
  --data khb_dataset/data.yaml \
  --epochs 100 \
  --imgsz 960 \
  --batch 1 \
  --name khb_field_model_6fields


# 5) Use trained model to get data from a new invoice image
# No label file is used here.
python main.py \
  --image_path images \
  --conf 0.25


# 6) If the model misses fields, try lower confidence
python main.py \
  --image_path images \
  --conf 0.10


python main.py \
  --image_path images \
  --conf 0.10 \
  --crops-dir crops


  python mainv2.py --image images --tesseract --conf 0.10 --crops-dir crops 