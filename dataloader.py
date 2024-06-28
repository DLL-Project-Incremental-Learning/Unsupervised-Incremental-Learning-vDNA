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
    def __init__(self, file_path, train_ratio=0.8, num_buckets=5):
        self.data = self.load_data(file_path)
        self.train_ratio = train_ratio
        self.num_buckets = num_buckets

    @staticmethod
    def load_data(file_path):
        with open(file_path) as f:
            return json.load(f)

    def split_data(self, buckets):
        train_buckets = []
        val_buckets = []

        for bucket in buckets:
            random.shuffle(bucket)
            split_index = int(self.train_ratio * len(bucket))
            train_buckets.append(bucket[:split_index])
            val_buckets.append(bucket[split_index:])
        
        return train_buckets, val_buckets

    def create_buckets(self, data, sort_key=None, reverse=False):
        if sort_key:
            data = sorted(data, key=lambda x: x[sort_key], reverse=reverse)
        
        bucket_size = len(data) // self.num_buckets
        buckets = [data[i * bucket_size:(i + 1) * bucket_size] for i in range(self.num_buckets)]

        # # Handle any remaining data points
        # remainder = len(data) % self.num_buckets
        # if remainder > 0:
        #     buckets[-1].extend(data[-remainder:])
        
        return buckets

    def asc_buckets(self):
        return self.split_data(self.create_buckets(self.data, sort_key='emd', reverse=False))

    def desc_buckets(self):
        return self.split_data(self.create_buckets(self.data, sort_key='emd', reverse=True))

    def random_buckets(self):
        random.shuffle(self.data)
        return self.split_data(self.create_buckets(self.data))


# Example usage:
# num_buckets = 6
# bucket_idx = 1
# processor = DataProcessor('results.json', num_buckets=num_buckets, train_ratio=0.8)
# train_buckets, val_buckets = processor.asc_buckets()
# image_files = [d['image'] for d in train_buckets[bucket_idx]]
# print("\n\nNumber of images: %d" % len(image_files))
# print("Image files: %s" % image_files[:5])
