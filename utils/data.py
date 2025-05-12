import os
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils.classes import load_class2idx_dict, load_idx2class_dict
from utils.preprocess import transform_annot

# Special Method는 파이썬 인터프리터를 위한 것
class PascalVoc2007Dataset(Dataset):
    def __init__(self, dataset_path, dataset_type='train', transform=None):
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.transform = transform
        self._set_paths()
        self._load_ids()
        
        
    def _set_paths(self):
        self.dataset_path = os.path.join(self.dataset_path, 'VOCdevkit', 'VOC2007')
        self.image_path = os.path.join(self.dataset_path, 'JPEGImages')
        self.annot_path = os.path.join(self.dataset_path, 'AnnotationsBoundingBoxes')
        self.id_path = os.path.join(self.dataset_path, 'TrainValTestIDs')
        
        
    def _load_ids(self):
        if self.dataset_type == 'train':
            id_file = os.path.join(self.id_path, 'train.txt')
        elif self.dataset_type == 'val':
            id_file = os.path.join(self.id_path, 'val.txt')
        elif self.dataset_type == 'test':
            id_file = os.path.join(self.id_path, 'test.txt')
            
        with open(id_file, 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]
    
    
    def __getitem__(self, sample_idx):
        id = self.ids[sample_idx]
        image_file = os.path.join(self.image_path, f'{id}.jpg')
        annot_file = os.path.join(self.annot_path, f'{id}.pt')
        
        image = Image.open(image_file)
        annot = torch.load(annot_file, weights_only=False)  # 보안 문제로 인한 weights_only=False 설정
        
        if self.transform:
            image, annot = self.transform(image, annot)
        
        return image, annot
    
    
    def __len__(self):
        return len(self.ids)
    

if __name__ == '__main__':
    dataset_path = '.'
    
    train_ds = PascalVoc2007Dataset(dataset_path, 'train')
    image, annot = train_ds[2]
    print(image.size)
    print(annot)
    
    # val_ds = PascalVoc2007Dataset(dataset_path, 'val')
    # test_ds = PascalVoc2007Dataset(dataset_path, 'test')