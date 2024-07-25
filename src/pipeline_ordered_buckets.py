from tqdm import tqdm
import network
import utils
import os
import sys
import random
import argparse
import numpy as np
import json

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils import data
# from .datasets import Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics
import network

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
from datasets.dataloaders import DataProcessor, KITTI360Dataset, DatasetLoader
from weaklabelgenerator import labelgenerator
from finetune_bucket import finetuner


# Define transforms
transform = transforms.Compose(
    [
        # transforms.Resize((512, 1024)),
        transforms.ToTensor()
    ]
)

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="cityscapes",
        choices=["cityscapes"],
        help="Name of dataset",
    )
    parser.add_argument(
        "--num_classes", type=int, default=19, help="num classes (default: None)"
    )

    # parser.add_argument("--bucketidx", type=int, required=True,
    #                     help="bucket index to use")
    parser.add_argument(
        "--buckets_order",
        type=str,
        default="asc",
        choices=["asc", "desc", "rand"],
        help="bucket order (asc, desc or rand) (default: asc)",
    )
    parser.add_argument(
        "--buckets_num", type=int, default=6, help="number of buckets (default: 6)"
    )
    parser.add_argument(
        "--overwrite_old_pred",
        action="store_true",
        default=False,
        help="overwrite old predictions",
    )

    # Deeplab Options
    available_models = sorted(
        name
        for name in network.modeling.__dict__
        if name.islower()
        and not (name.startswith("__") or name.startswith("_"))
        and callable(network.modeling.__dict__[name])
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deeplabv3plus_resnet101",
        choices=available_models,
        help="model name",
    )
    parser.add_argument(
        "--separable_conv",
        action="store_true",
        default=False,
        help="apply separable conv to decoder and aspp",
    )
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument(
        "--save_val_results",
        action="store_true",
        default=False,
        help='save segmentation results to "./results"',
    )
    parser.add_argument(
        "--total_itrs", type=int, default=1800, help="epoch number (default: 30k)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--lr_policy",
        type=str,
        default="poly",
        choices=["poly", "step"],
        help="learning rate scheduler policy",
    )
    parser.add_argument("--step_size", type=int, default=600)
    parser.add_argument(
        "--crop_val",
        action="store_true",
        default=False,
        help="crop validation (default: False)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size (default: 16)"
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=4,
        help="batch size for validation (default: 4)",
    )

    # parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--crop_size", type=int, default=370)

    parser.add_argument(
        "--ckpt", default=None, type=str, help="restore from checkpoint"
    )
    parser.add_argument("--continue_training", action="store_true", default=False)

    parser.add_argument(
        "--loss_type",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "focal_loss"],
        help="loss type (default: False)",
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="weight decay (default: 1e-4)"
    )
    parser.add_argument(
        "--random_seed", type=int, default=1, help="random seed (default: 1)"
    )
    parser.add_argument(
        "--print_interval",
        type=int,
        default=10,
        help="print interval of loss (default: 10)",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=250,
        help="epoch interval for eval (default: 100)",
    )
    parser.add_argument(
        "--download", action="store_true", default=False, help="download datasets"
    )

    parser.add_argument(
        "--json_input", type=str, default="rank_1_val.json",
    )

    parser.add_argument(
        "--layer",
        type=str,
        default="l1",
        help="layer number",
        choices=["l1", "l2", "l3", "l4", "l5", "sl", "gl"],
    )
    parser.add_argument(
        "--full", type=str, default="True", help="full or Bias BN", choices=["True", "False"]
    )
    parser.add_argument("--kd", type=str, default="False", help="Knowledge distillation")
    parser.add_argument(
        "--pixel", type=str, default="False", help="Pixel level distillation"
    )

    return parser


def get_n_image_paths(json_file, n):
    # Read the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    # Extract all image paths
    image_paths = [item["image"] for item in data]

    # If N is greater than the number of available paths, return all paths
    if n >= len(image_paths):
        return image_paths

    # Otherwise, return N random paths
    return random.sample(image_paths, n)


def main():

    opts = get_argparser().parse_args()
    opts.dataset = "KITTI-360-FILTERED"
    use_pixel = opts.pixel
    random.seed(opts.random_seed)

    # forcefully delete outputs folder
    # os.system("rm -rf outputs")

    num_buckets = 1
    input_file = os.path.join("assets", opts.json_input)
    print(f"Using input file: {input_file}")

    processor = DataProcessor(
        input_file, num_buckets=num_buckets, train_ratio=0.8
    )

    # Dictionary to map bucket orders to their respective methods
    bucket_methods = {
        "asc": processor.asc_buckets,
        "desc": processor.desc_buckets,
        "rand": processor.random_buckets,
    }
    # Fetch the appropriate method based on the bucket_order
    bucket_method = bucket_methods.get(opts.buckets_order)
    # Error handling for invalid bucket_order
    if bucket_method is None:
        raise ValueError(
            f"Invalid bucket_order: {opts.buckets_order}, used ascendant order instead"
        )

    train_buckets = bucket_method()
    # val_data = processor.val_data

    model_name = "deeplabv3plus_resnet101"
    ckpt = "./checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth"
    teacher_ckpt = ckpt
    teacher_model = network.modeling.__dict__[model_name](
        num_classes=19, output_stride=16
    )
    model = network.modeling.__dict__[model_name](num_classes=19, output_stride=16)

    # for bucket_idx in range(num_buckets):
    bucket_idx = 0
    print("\n\n[INFO] Bucket %d" % bucket_idx)

    image_files = [d["image"] for d in train_buckets[bucket_idx]]
    samples = random.sample(image_files, 2000)

    print("\n\n[INFO] Generating weak labels for bucket %d" % bucket_idx)
    filtered_samples, train_labelgen = labelgenerator(
        samples, model, ckpt, bucket_idx, val=False, order=opts.buckets_order, use_pixel=use_pixel
    )
    
    # filtered_samples = samples
    # train_labelgen = "outputs/weaklabels/KITTI-360/rand/bucket_0/"

    print("\n\n[INFO] Starting finetuning for bucket %d" % bucket_idx)

    if bucket_idx >= 0:
        finetuner(
            opts=opts,
            model=model,
            teacher_model=teacher_model,
            teacher_ckpt=teacher_ckpt,
            checkpoint=ckpt,
            bucket_idx=bucket_idx,
            train_image_paths=filtered_samples,
            train_label_dir=train_labelgen,
            model_name=model_name,
        )

    # ckpt = "./checkpoints/latest_bucket_%s_%s_%s_%s_os%d.pth" % (
    #     bucket_idx,
    #     opts.buckets_order,
    #     model_name,
    #     "kitti",
    #     opts.output_stride,
    # )

    print(f"Iteration completed.")
    print(
        "\n------------------------------------------------------------------------------------------------------------------------\n"
    )


if __name__ == "__main__":
    main()
