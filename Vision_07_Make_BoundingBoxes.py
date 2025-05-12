import torch
import os
import random
from torchvision.tv_tensors import BoundingBoxes
from utils.classes import load_class2idx_dict
from xml.etree import ElementTree as ET

def make_bounding_boxes(annot_files, class2idx):
    for annot_file in annot_files:
        # XML 파일 파싱으로 bounding box 정보 추출
        annot_file_path = os.path.join(annot_path, annot_file)
        tree = ET.parse(annot_file_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in class2idx:
                print(f"Class {name} not found in class2idx.")
                continue
            
            size = root.find('size')
            H = int(size.find('height').text)
            W = int(size.find('width').text)
            
            # bounding box 좌표 추출
            bbox = obj.find('bndbox')
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)
            
            bboxes.append([x_min, y_min, x_max, y_max])
            labels.append(class2idx[name])
            
        # BoundingBoxes 객체 생성
        if bboxes:
            bboxes = BoundingBoxes(bboxes, format='xyxy', canvas_size=(H, W))
            labels = torch.tensor(labels)
            target = {'boxes': bboxes, 'labels': labels}
            
            # 파일명.pt로 annot_new_path에 저장
            file_id = annot_file.split('.')[0]
            save_path = os.path.join(annot_box_path, f"{file_id}.pt")
            torch.save(target, save_path)
            print(f"Saved bounding boxes for {annot_file} to {save_path}")
        else:
            print(f"No bounding boxes found for {annot_file}")
        

dataset_path = os.path.join('VOCdevkit', 'VOC2007')
annot_path = os.path.join(dataset_path, 'Annotations')
annot_box_path = os.path.join(dataset_path, 'AnnotationsBoundingBoxes')
os.makedirs(annot_box_path, exist_ok=True)

annot_files = [annot_file for annot_file in os.listdir(annot_path) if annot_file.endswith('.xml')]

class2idx = load_class2idx_dict('.')
class_names = list(class2idx.keys())

make_bounding_boxes(annot_files, class2idx)

# 검증
path = os.path.join(annot_box_path, '000991.pt')
target = torch.load(path, weights_only=False)
print(target)