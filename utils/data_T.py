import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T2

class PascalVoc2007Dataset(Dataset):
    def __init__(self, dataset_path, dataset_type='train'):
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self._set_paths()
        self._load_ids()
        self._set_transform()
    
    def _set_paths(self):
        self.dataset_path = os.path.join(self.dataset_path, 'VOCdevkit', 'VOC2007')
        self.image_path = os.path.join(self.dataset_path, 'JPEGImages')
        self.annot_path = os.path.join(self.dataset_path, 'AnnotationsBoundingBoxes')
        self.id_path = os.path.join(self.dataset_path, 'TrainValTestIDs')

    def _load_ids(self):
        id_file = os.path.join(self.id_path, f'{self.dataset_type}.txt')
        with open(id_file, 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]
    
    def _set_transform(self):
        if self.dataset_type == 'train':
            self.transform = T2.Compose([
                T2.ToImage(),
                T2.Resize(size=(448, 448)),
                T2.RandomHorizontalFlip(1),
                T2.RandomAffine(translate=(0.2, 0.2), scale=(0.8, 1.2), degrees=0),
            ])
        else:
            self.transform = T2.Compose([
                T2.ToImage(),
                T2.Resize(size=(448, 448)),
            ])
    
    def __getitem__(self, sample_idx):
        id = self.ids[sample_idx]
        image_file = os.path.join(self.image_path, f'{id}.jpg')
        annot_file = os.path.join(self.annot_path, f'{id}.pt')

        image = Image.open(image_file).convert("RGB")
        annot = torch.load(annot_file, weights_only=False)

        if self.transform:
            image, annot = self.transform(image, annot)  # ✅ 내부 transform 사용

        return image, annot

    def __len__(self):
        return len(self.ids)
