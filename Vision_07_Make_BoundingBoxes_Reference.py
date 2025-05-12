import os
import torch
from utils.classes import load_class2idx_dict
from utils.preprocess import transform_annot
import config

dataset_path = config.dataset_path
dataset_path = os.path.join(dataset_path, 'VOCdevkit', 'VOC2007')

annot_path = os.path.join(dataset_path, 'Annotations')
annot_new_path = os.path.join(dataset_path, 'AnnotationsBoundingBoxes')

os.makedirs(annot_new_path, exist_ok=True)

class2idx = load_class2idx_dict(config.dataset_path)
annot_files = [annot_file for annot_file in os.listdir(annot_path) if annot_file.endswith('.xml')]

for annot_file in annot_files:
    save_file = annot_file.replace('.xml', '.pt')
    annot_file = os.path.join(annot_path, annot_file)
    
    target = transform_annot(annot_file, class2idx)
    torch.save(target, os.path.join(annot_new_path, save_file))