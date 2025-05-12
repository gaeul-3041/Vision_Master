from utils.data import PascalVoc2007Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def draw_bbox(ax, img, annot):
    ax.imshow(img)
    ax.axis('off')

    bboxes = annot['boxes']
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        rect = Rectangle((xmin, ymin), width=xmax-xmin, height=ymax-ymin, facecolor='none', edgecolor='r')
        ax.add_patch(rect)

train_ds = PascalVoc2007Dataset('.', 'train')
fig, axes = plt.subplots(2, 2, dpi=200, layout='constrained')

for ax_idx, ax in enumerate(axes.flat):
    img, annot = train_ds[ax_idx]
    draw_bbox(ax, img, annot)

plt.show()