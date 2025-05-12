import os
import json


def load_class2idx_dict(dataset_path):
    dataset_path = os.path.join(dataset_path, 'VOCdevkit', 'VOC2007')
    with open(os.path.join(dataset_path, 'class2idx.json'), 'r') as f:
        class2idx_dict = json.load(f)
    return class2idx_dict


def load_idx2class_dict(dataset_path):
    dataset_path = os.path.join(dataset_path, 'VOCdevkit', 'VOC2007')
    with open(os.path.join(dataset_path, 'idx2class.json'), 'r') as f:
        idx2class_dict = json.load(f)
    return idx2class_dict