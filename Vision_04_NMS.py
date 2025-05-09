from PIL import Image
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt

from utils import get_iou_mat

img = Image.open("bus.jpg")
model = YOLO("yolo11n.pt")
pred = model.predict(img, iou=0.98)

bboxes = pred[0].boxes.xyxy
conf = pred[0].boxes.conf
iou = get_iou_mat(bboxes)

# NMS
def non_maximum_suppression(bboxes, conf, iou, iou_threshold=0.5):
    indices = torch.arange(len(bboxes))
    keep = []

    while len(indices) > 0:
        conf_subset = conf[indices]
        max_conf_idx = conf_subset.argmax()
        max_index = indices[max_conf_idx]

        keep.append(max_index.item())

        if len(indices) == 1:
            break

        iou_row = iou[max_index][indices]
        indices = indices[iou_row < iou_threshold]

    return keep

keep_indices = non_maximum_suppression(bboxes, conf, iou)
print(keep_indices)
print(bboxes[keep_indices])