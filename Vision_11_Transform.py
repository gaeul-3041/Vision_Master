import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms.v2 as T2 

from utils.data import PascalVoc2007Dataset
from utils.vis import draw_bbox

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

transform = T2.Compose([
    T2.ToImage(),
    T2.Resize(size=(448, 448)),
    T2.RandomHorizontalFlip(1),  # Augmentation 실제 과정 확인
    T2.RandomAffine(translate=(0.2, 0.2), scale=(0.8, 1.2), degrees=0)
    # T2.CenterCrop(size=(448, 448))
])

train_ds = PascalVoc2007Dataset('.', 'train')
fig, axes = plt.subplots(1, 2, dpi=200, layout='constrained')

img, annot = train_ds[5]
img_t, annot_t = transform(img, annot)
draw_bbox(axes[0], img, annot)
draw_bbox(axes[1], img_t.permute(1, 2, 0), annot_t)

plt.show()

# matplotlib은 imread로 (H, W, 3)을 읽어오고, color channel은 RGB 순
# OpenCV는 BGR로 읽어옴
# torch는 (3, H, W)로 읽어옴