import json
import random
import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import network
import utils
from datasets import Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from collections import namedtuple
import wandb

class DataProcessor:
    def __init__(self, file_path, train_ratio=0.8, num_buckets=6, seed=42, dist_buckets=-1):
        self.data = self.load_data(file_path)
        self.train_ratio = train_ratio
        self.num_buckets = num_buckets
        self.train_data, self.val_data = self.split_data(self.data)
        self.dist_buckets = dist_buckets
        random.seed(seed)

    @staticmethod
    def load_data(file_path):
        with open(file_path) as f:
            return json.load(f)
    
    def split_data(self, data):

        train_data = [d for d in data if not d['val_set']]
        val_data = [d for d in data if d['val_set']]

        return train_data, val_data

    def create_buckets(self, data, sort_key=None, reverse=False):
        if sort_key:
            data = sorted(data, key=lambda x: x[sort_key], reverse=reverse)
        
        if self.dist_buckets < 0:
            bucket_size = len(data) // self.num_buckets
            buckets = [data[i * bucket_size:(i + 1) * bucket_size] for i in range(self.num_buckets)]
        else:
            buckets = []
            j = 0
            for i in range(len(data)):
                if abs(data[i]['emd'] - data[j]['emd']) > self.dist_buckets:
                    buckets.append(data[j:i+1])
                    j = i+1
            buckets.append(data[j:])

        # # Handle any remaining data points
        # remainder = len(data) % self.num_buckets
        # if remainder > 0:
        #     buckets[-1].extend(data[-remainder:])
        
        return buckets

    def asc_buckets(self):
        return self.create_buckets(self.train_data, sort_key='emd', reverse=False)

    def desc_buckets(self):
        return self.create_buckets(self.train_data, sort_key='emd', reverse=True)

    def random_buckets(self):
        random.shuffle(self.train_data)
        return self.create_buckets(self.train_data)


# Custom dataset for KITTI360 with weak labels
class KITTI360Dataset(Dataset):

    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])



    def __init__(self, image_paths, label_dir, transform=None):
        # self.image_paths = image_paths
        self.image_paths = image_paths
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        target_path = os.path.join(self.label_dir, os.path.basename(img_path))
        image = Image.open(img_path).convert("RGB")
        target = Image.open(target_path)
        # print(f"img_path: {img_path}")
        # print(f"label_dir: {os.path.join(self.label_dir, os.path.basename(img_path))}")

        if self.transform:
            # image = self.transform(image)
            # target = self.transform(target)
            # target = torch.squeeze(target, 0)
            image, target = self.transform(image, target)
        return image, target

class DatasetLoader:
    def __init__(self, opts):
        self.opts = opts

    def get_transforms(self):
        train_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(self.opts.crop_size, self.opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        return train_transform, val_transform

    def get_datasets(self, train_image_paths, train_label_dir):
        train_transform, val_transform = self.get_transforms()

        train_dst = KITTI360Dataset(
            image_paths=train_image_paths,
            label_dir=train_label_dir,
            transform=train_transform
        )

        # val_dst = KITTI360Dataset(
        #     image_paths=val_image_paths,
        #     label_dir=val_label_dir,
        #     transform=val_transform
        # )

        return train_dst #, val_dst


# Example usage:
# num_buckets = 6
# bucket_idx = 1
# processor = DataProcessor('results.json', num_buckets=num_buckets, train_ratio=0.8)
# train_buckets, val_buckets = processor.asc_buckets()
# image_files = [d['image'] for d in train_buckets[bucket_idx]]
# print("\n\nNumber of images: %d" % len(image_files))
# print("Image files: %s" % image_files[:5])