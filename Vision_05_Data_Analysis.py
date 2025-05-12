import os
import json

dataset_path = os.path.join('VOCdevkit', 'VOC2007')
annot_path = os.path.join(dataset_path, 'Annotations')
image_path = os.path.join(dataset_path, 'JPEGImages')

# img_files = os.listdir(image_path)를 바로 써버리면 .DS_Store 같은 파일도 포함, 확장자 반드시 검증
img_files = [img_file for img_file in os.listdir(image_path) if img_file.endswith('.jpg')]
annot_files = [annot_file for annot_file in os.listdir(annot_path) if annot_file.endswith('.xml')]

print(f'Number of images: {len(img_files)}')
print(f'Number of annotations: {len(annot_files)}')

# file_ids = []
# for file_idx in range(1, len(img_files) + 1):
#    file_ids.append(f"{file_idx:06d}")   
file_ids = [f"{file_idx:06d}" for file_idx in range(1, len(img_files) + 1)]
annot_ids = [f"{file_id}.xml" for file_id in file_ids]
img_ids = [f"{file_id}.jpg" for file_id in file_ids]

# 중복 제거
img_files = set([img_file.split('.')[0] for img_file in img_files])
annot_files = set([annot_file.split('.')[0] for annot_file in annot_files])
print(img_files == annot_files)

# class2idx.json, idx2class.json 생성
class2idx, idx2class = {}, {}
class_path = os.path.join(dataset_path, 'ImageSets', 'Main')
class_files = sorted(os.listdir(class_path))
class_names = []

for class_file in class_files:
    if '_' in class_file:
        class_name = class_file.split('_')[0]
        if class_name not in class_names:
            class_names.append(class_name)
        
for idx, class_name in enumerate(class_names):
    class2idx[class_name] = idx
    idx2class[idx] = class_name
    
print(f'Number of classes: {len(class_names)}')
print(f'Class to index mapping: {class2idx}')
print(f'Index to class mapping: {idx2class}')

# JSON 파일로 저장
with open(os.path.join(dataset_path, 'class2idx.json'), 'w') as f:
    json.dump(class2idx, f, indent=4)
    
with open(os.path.join(dataset_path, 'idx2class.json'), 'w') as f:
    json.dump(idx2class, f, indent=4)
    
# 주의: idx2class의 경우 key가 str으로 변환되어 저장, 불러올 때 주의할 것