import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def draw_bbox(ax, img, annot):
    ax.imshow(img)
    ax.axis('off')

    bboxes = annot['boxes']
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        rect = Rectangle((xmin, ymin), width=xmax-xmin, height=ymax-ymin, facecolor='none', edgecolor='r')
        ax.add_patch(rect)