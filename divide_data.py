"""
数据集位置：E:\cat_dog_classify\cat_dog
数据个数：12000张图片、猫狗各6000张
数据集形式：type.num.jpeg
初步定train_test_valid的划分比例为：6-2-2
最终得到的组织形式：
cat_dog_datasets
            --train(7200)
            --test(2400)
            --valid(2400)
"""
import os
import shutil
import numpy as np
import collections

def copyfile(filename, target_dir):
    """复制文件filename到指定路径target_dir"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def split_train_test_valid(src_dir=r"E:\cat_dog_classify", test_ratio=0.2, valid_ratio=0.2, target_dir=r"E:\cat_dog_classify\cat_dog_datasets"):
    for type in ['train', 'test', 'valid']:
        if not os.path.exists(os.path.join(target_dir, type)):
            os.makedirs(os.path.join(target_dir, type))
    n = 6000
    n_valid_per_label = valid_ratio * n
    n_test_per_label = test_ratio * n
    label_count = {} 
    #label只有两种'0'和'1'
    for file in os.listdir(os.path.join(src_dir, 'cat_dog')):
        label = file.split('.')[0]
        fname = os.path.join(src_dir, 'cat_dog', file)
        if label not in label_count or label_count[label]<n_valid_per_label:
            copyfile(fname, os.path.join(target_dir, 'valid'))
            label_count[label]=label_count.get(label, 0) + 1
        elif label_count[label]<n_valid_per_label+n_test_per_label:
            copyfile(fname, os.path.join(target_dir, 'test'))
            label_count[label]=label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(target_dir, 'train'))
    return n_valid_per_label, n_test_per_label

if __name__=='__main__':
    split_train_test_valid()

