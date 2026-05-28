# KHB YOLO Dataset

This folder is for training a YOLOv8 model to detect delivery-order fields.

## Classes

```text
0 Invoice No
1 Invoice Date
2 Dealer Code
3 Sale Order
4 Vender Code
5 Vehicle Code
6 Route
7 Warehouse
```

## Sample

The starter sample is:

```text
train/images/khb_sample_001.jpeg
train/labels/khb_sample_001.txt
```

Each label line uses YOLO format:

```text
class_id x_center y_center width height
```

All coordinates are normalized from `0` to `1`.

When you add your own images, make sure the image and label filenames match:

```text
train/images/invoice_002.jpeg
train/labels/invoice_002.txt
```
