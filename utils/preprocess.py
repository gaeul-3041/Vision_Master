from xml.etree import ElementTree as ET
import torch
from torchvision.tv_tensors import BoundingBoxes

def transform_annot(annot_file, class2idx):
    tree = ET.parse(annot_file)
    root = tree.getroot()
    
    size = root.find('size')
    W = int(size.find('width').text)
    H = int(size.find('height').text)
    
    objects = root.findall('object')
    labels, bboxes = [], []
    
    for object_ in objects:
        class_ = object_.find('name').text
        labels.append(class_)
        
        bbox = object_.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        bbox = [xmin, ymin, xmax, ymax]
        bboxes.append(bbox)
        
    bboxes = BoundingBoxes(bboxes, format='XYXY',canvas_size=(H, W))
    labels = torch.tensor([class2idx[class_] for class_ in labels])
    target = {'boxes': bboxes, 'labels': labels}
    
    return target