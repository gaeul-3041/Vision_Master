from PIL import Image
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt

from utils.iou import get_iou_mat

img = Image.open("bus.jpg")
model = YOLO("yolo11n.pt")
pred = model.predict(img, iou=0.98)

bboxes = pred[0].boxes.xyxy
conf = pred[0].boxes.conf
iou = get_iou_mat(bboxes)

threshold = 0.5
n_bboxes = len(bboxes)
keep = torch.ones(n_bboxes, dtype=torch.bool)

# 바운딩 박스가 적은 편이라 단순 for문 구현
for bbox_idx in range(n_bboxes):
    if bool(keep[bbox_idx]) is True:
        iou_ = iou[bbox_idx]
        iou_high = torch.where(iou_ > threshold)[0]
        keep[iou_high] = False  # 이것만 쓰면 스스로를 죽여버리는 문제 발생
        keep[bbox_idx] = True  # 자기 자신은 살림
        
bboxes_nms = bboxes[keep]  # boolean indexing은 True인 것만 가져옴
conf_nms = conf[keep]

print(bboxes_nms)
print(conf_nms)
print(keep)