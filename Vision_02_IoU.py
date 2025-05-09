from PIL import Image
from ultralytics import YOLO

img = Image.open("bus.jpg")
model = YOLO("yolo11n.pt")
pred = model.predict(img, iou=0.98)

pred = pred[0]
bboxes = pred.boxes.xyxy

bbox0, bbox1 = bboxes[2], bboxes[3]
print(bbox0, bbox1, '\n')

# IoU 계산
def get_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = b1_area + b2_area - intersection

    # return intersection / (union + 1e-6)
    return intersection / union if union > 0 else 0.0

print("IoU:", get_iou(bbox0, bbox1))