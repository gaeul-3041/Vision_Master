import torch
import os
import config
from torchvision.tv_tensors import BoundingBoxes
from utils.classes import load_class2idx_dict

H, W = 500, 353
bboxes = [[48, 240, 195, 371], [8, 12, 352, 498]]
labels = ['dog', 'person']

# H, W 순서 주의 - 여기서는 H 먼저
bboxes = BoundingBoxes(bboxes, format='xyxy', canvas_size=(H, W))
print(bboxes)

class2idx = load_class2idx_dict('.')
labels = torch.tensor([class2idx[label] for label in labels])
print(labels)

target = {'boxes': bboxes, 'labels': labels}
# torch.save(target, '000001.pt')
print(target)