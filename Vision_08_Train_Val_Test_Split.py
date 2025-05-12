import os
import random
import config
import numpy as np

dataset_path = config.dataset_path
dataset_path = os.path.join(dataset_path, 'VOCdevkit', 'VOC2007')
annot_path = os.path.join(dataset_path, 'Annotations')
ids = [annot_file.split('.')[0] for annot_file in os.listdir(annot_path) if annot_file.endswith('.xml')]

# train / val / test index 저장하는 txt 파일 생성
os.makedirs(os.path.join(dataset_path, 'TrainValTestIDs'), exist_ok=True)

n_total_data = len(ids)
n_train_data = int(n_total_data * config.train_ratio)
n_val_data = int(n_total_data * config.val_ratio)
n_test_data = n_total_data - n_train_data - n_val_data

# np.random.seed, normal, uniform, shuffle 등은 권장하지 않음
rng = np.random.default_rng(0)
random_indices = np.arange(n_total_data)
rng.shuffle(random_indices)

ids = np.array(ids)
train_ids = ids[random_indices[:n_train_data]]
val_ids = ids[random_indices[n_train_data:-n_test_data]]
test_ids = ids[random_indices[-n_test_data:]]

with open(os.path.join(dataset_path, 'TrainValTestIDs', 'train.txt'), 'w') as f:
    for id in train_ids:
        f.write(id + '\n')

with open(os.path.join(dataset_path, 'TrainValTestIDs', 'val.txt'), 'w') as f:
    for id in val_ids:
        f.write(id + '\n')

with open(os.path.join(dataset_path, 'TrainValTestIDs', 'test.txt'), 'w') as f:
    for id in test_ids:
        f.write(id + '\n')


            
# # 검증을 위해 train.txt, val.txt, test.txt 파일을 읽어서 개수 출력
# with open(os.path.join(dataset_path, 'TrainValTestIDs', 'train.txt'), 'r') as f:
#     train_ids = f.readlines()
# print(f"Number of train IDs: {len(train_ids)}")
# with open(os.path.join(dataset_path, 'TrainValTestIDs', 'val.txt'), 'r') as f:
#     val_ids = f.readlines()
# print(f"Number of val IDs: {len(val_ids)}")
# with open(os.path.join(dataset_path, 'TrainValTestIDs', 'test.txt'), 'r') as f:
#     test_ids = f.readlines()
# print(f"Number of test IDs: {len(test_ids)}")

# print(test_ids)

# # 모든 ID를 합쳤을 때의 개수 출력(중복 제거)
# print(f"Total number of unique IDs: {len(set(train_ids + val_ids + test_ids))}")