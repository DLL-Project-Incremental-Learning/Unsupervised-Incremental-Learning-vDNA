from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
import wandb

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from torchvision import transforms
# from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader, Dataset
from glob import glob
from collections import namedtuple
from dataloaders import DataProcessor, KITTI360Dataset, DatasetLoader
from weaklabelgenerator import labelgenerator
import network

# Define transforms
transform = transforms.Compose([
    # transforms.Resize((512, 1024)),
    transforms.ToTensor()
])

from finetune_bucket import finetuner


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=19,
                        help="num classes (default: None)")
    
    # parser.add_argument("--bucketidx", type=int, required=True,
    #                     help="bucket index to use")
    parser.add_argument("--buckets_order", type=str, default='asc',
                        choices=['asc', 'desc', 'rand'],
                        help="bucket order (asc, desc or rand) (default: asc)")
    parser.add_argument("--buckets_num", type=int, default=6,
                        help="number of buckets (default: 6)")
    parser.add_argument("--overwrite_old_pred", action='store_true', default=False,
                        help="overwrite old predictions")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=20,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    # parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--crop_size", type=int, default=189)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=10,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    return parser


def main():
        
    num_buckets = 6
    processor = DataProcessor('results.json', num_buckets=num_buckets, train_ratio=0.8)
    train_buckets = processor.asc_buckets()
    # val_data = processor.val_data
    model_name = 'deeplabv3plus_resnet101'
    ckpt = "checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth"
    model = network.modeling.__dict__[model_name](num_classes=19, output_stride=16)


    opts = get_argparser().parse_args()
    opts.dataset = 'cityscapes'


    for bucket_idx in range(num_buckets):
        print("\n\n[INFO] Bucket %d" % bucket_idx)

        image_files = [d['image'] for d in train_buckets[bucket_idx]]
        # val_image_files = [d['image'] for d in val_buckets[bucket_idx]]
        # print("\n\nNumber of images: %d" % len(image_files[1]))
        # print("Image files: %s" % image_files[:4])

        samples = image_files[:20]
        # val_samples = val_image_files[:10]
        # print("\n\nSamples: %s" % samples)
        # print("Validation Samples: %s" % val_samples)

        print("\n\n[INFO] Generating weak labels for bucket %d" % bucket_idx)
        train_labelgen = labelgenerator(samples, model, ckpt, bucket_idx, val=False, order="asc")
        # val_labelgen = labelgenerator(val_samples, model, ckpt, bucket_idx, val=True, order="asc")

        print("\n\n[INFO] Starting finetuning for bucket %d" % bucket_idx)
        finetuner(opts=opts, model=model, checkpoint=ckpt, bucket_idx=bucket_idx, train_image_paths=samples, train_label_dir=train_labelgen, model_name=model_name)

        ckpt = 'checkpoints/latest_bucket_%s_asc_%s_%s_os%d.pth' % (bucket_idx, model_name, "kitti", opts.output_stride)
        
        print("\n\n[INFO] Loading model from checkpoint %s" % ckpt)
        print(f"Iteration {bucket_idx} completed. Moving to next bucket...")
        print("\n------------------------------------------------------------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()