import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import config as cfg



devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')
data_path = os.path.join(devkil_path, 'VOC2007')
cache_path = cfg.CACHE_PATH   #
batch_size = cfg.BATCH_SIZE   #  1
image_size = cfg.IMAGE_SIZE   #  448
cell_size = cfg.CELL_SIZE     #  7
classes = cfg.CLASSES         # 20个类
class_to_ind = dict(zip(classes, range(len(classes))))  # 类的 IND
flipped = cfg.FLIPPED  # True
phase = 'Train'  # Train
rebuild = False # False
cursor = 0
epoch = 1
gt_labels = None



cache_file = os.path.join(cache_path, 'pascal_' + phase + '_gt_labels.pkl')  #   地址

if os.path.isfile(cache_file) and not rebuild:
    print('Loading gt_labels from: ' + cache_file)
with open(cache_file, 'rb') as f:
    gt_labels = pickle.load(f)

print(gt_labels[0]['label'][3,4,0:25])
if flipped:
    print('Appending horizontally-flipped training examples ...')
    gt_labels_cp = copy.deepcopy(gt_labels)   #字典的深度拷贝 改变数据却不改变原数据
    print(gt_labels_cp[0]['label'][3,4,0:25])
    for idx in range(len(gt_labels_cp)):
        gt_labels_cp[idx]['flipped'] = True
        gt_labels_cp[idx]['label'] =\
            gt_labels_cp[idx]['label'][:, ::-1, :]
        print(gt_labels_cp[idx]['label'][3,2,0:25])  #索引调换
        for i in range(cell_size):
            for j in range(cell_size):
                if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                    # print(gt_labels_cp[idx]['label'][i, j, 1])
                    gt_labels_cp[idx]['label'][i, j, 1] = \
                        image_size - 1 -\
                        gt_labels_cp[idx]['label'][i, j, 1]
                    # print(gt_labels_cp[idx]['label'][i, j, 1])
    print(gt_labels_cp[0]['label'][3, 2, 0:25])
    gt_labels += gt_labels_cp
    print(gt_labels[0]['label'][3, 6, 0:25])
print(1)

np.random.shuffle(gt_labels)
gt_labels = gt_labels
