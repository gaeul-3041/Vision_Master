from PIL import Image
from ultralytics import YOLO
import torch

img = Image.open("bus.jpg")
model = YOLO("yolo11n.pt")
pred = model.predict(img)

pred = pred[0]
bboxes = pred.boxes.xyxy
print(bboxes, '\n')

def xyxy_to_xywh(bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]

def xyxy_to_cxcywh(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

def xywh_to_xyxy(bbox):
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]

def xywh_to_cxcywh(bbox):
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    return [cx, cy, w, h]

def cxcywh_to_xyxy(bbox):
    cx, cy, w, h = bbox
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]

def cxcywh_to_xywh(bbox):
    cx, cy, w, h = bbox
    x1 = cx - w / 2
    y1 = cy - h / 2
    return [x1, y1, w, h]

bbox = bboxes[0]
print("Original bbox:", bbox)
print("xyxy to xywh:", xyxy_to_xywh(bbox))
print("xyxy to cxcywh:", xyxy_to_cxcywh(bbox))

bbox_xywh = xyxy_to_xywh(bbox)
print("xywh to xyxy:", xywh_to_xyxy(bbox_xywh))
print("xywh to cxcywh:", xywh_to_cxcywh(bbox_xywh))

bbox_cxcywh = xyxy_to_cxcywh(bbox)
print("cxcywh to xyxy:", cxcywh_to_xyxy(bbox_cxcywh))
print("cxcywh to xywh:", cxcywh_to_xywh(bbox_cxcywh))