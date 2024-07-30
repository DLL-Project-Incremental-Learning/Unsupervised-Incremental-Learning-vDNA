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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils import data
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
from torch.utils.data import DataLoader, Dataset
from glob import glob
from collections import namedtuple
from datasets.dataloaders import DataProcessor, KITTI360Dataset, DatasetLoader
from weaklabelgenerator import labelgenerator
from finetune_bucket import finetuner

def load_config(config_path):
    """
    Loads the configuration from a JSON file.

    Args:
        config_path (str): Path to the JSON configuration file.

    Returns:
        dict: The configuration as a dictionary.
    """
    with open(config_path, "r") as f:
        return json.load(f)

def main(config_path):
    """
    Main function to run the training and fine-tuning pipeline with ordered buckets.

    This function performs the following steps:
    1. Loads the configuration from a JSON file.
    2. Sets up the data processor and bucket ordering methods.
    3. Initializes the model and teacher model.
    4. Iteratively processes each bucket to generate weak labels and fine-tune the model.

    Args:
        config_path (str): Path to the JSON configuration file.
    """

    config = load_config(config_path)
    random.seed(config["random_seed"])

    # Initialize the data processor
    processor = DataProcessor(
        config["data_processor"]["json_file"],
        num_buckets=config["data_processor"]["num_buckets"],
        train_ratio=config["data_processor"]["train_ratio"],
    )

    # Dictionary to map bucket orders to their respective methods
    bucket_methods = {
        "asc": processor.asc_buckets,
        "desc": processor.desc_buckets,
        "rand": processor.random_buckets,
    }

    # Fetch the appropriate method based on the bucket_order
    bucket_method = bucket_methods.get(config["buckets_order"])

    # Error handling for invalid bucket_order
    if bucket_method is None:
        raise ValueError(f"Invalid bucket_order: {config['buckets_order']}")

    # Get the training buckets
    train_buckets = bucket_method()
    model_name = config["model"]

    ckpt = config["ckpt"]
    teacher_ckpt = config["teacher_ckpt"]

    # Initialize the model and teacher model
    model = network.modeling.__dict__[model_name](
        num_classes=config["num_classes"], output_stride=config["output_stride"]
    )
    teacher_model = network.modeling.__dict__[model_name](
        num_classes=config["num_classes"], output_stride=config["output_stride"]
    )

    # Process each bucket iteratively
    for bucket_idx in range(config["buckets_num"]):
        print(f"\n\n[INFO] Bucket {bucket_idx}")
        image_files = [d["image"] for d in train_buckets[bucket_idx]]
        samples = random.sample(image_files, config["labelgenerator"]["num_samples"])

        print(f"\n\n[INFO] Generating weak labels for bucket {bucket_idx}")
        filtered_samples, train_labelgen = labelgenerator(
            samples, model, ckpt, bucket_idx, val=False, order=config["buckets_order"]
        )

        print("\n\n[INFO] Starting finetuning for bucket %d" % bucket_idx)

        if bucket_idx >= 0:
            finetuner(
                opts=config,
                model=model,
                checkpoint=ckpt,
                teacher_model=teacher_model,
                teacher_ckpt=teacher_ckpt,
                bucket_idx=bucket_idx,
                train_image_paths=filtered_samples,
                train_label_dir=train_labelgen,
                model_name=model_name,
            )
        ckpt = f"./checkpoints/latest_bucket_{bucket_idx}_{config['buckets_order']}_{model_name}_kitti_os{config['output_stride']}.pth"

        print(f"Iteration {bucket_idx} completed.")
        print("\n" + "-" * 120 + "\n")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python pipeline_ordered_buckets.py <path_to_config.json>")
        sys.exit(1)
    main(sys.argv[1])
