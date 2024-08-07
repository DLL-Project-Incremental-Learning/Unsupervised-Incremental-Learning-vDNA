import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T
from typing import List, Dict, Tuple
from datasets import Cityscapes
import numpy as np
import json
import shutil

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the config file
config_path = "./configs/weak_label_kitti_360.json"

# Load configuration from JSON file
with open(config_path, "r") as f:
    config = json.load(f)

# Constants from config
CONFIDENCE_THRESHOLD = config["confidence_threshold"]
ENTROPY_THRESHOLD = config["entropy_threshold"]
BASE_SOURCE_DIR = config["base_source_dir"]
WEAK_LABEL_DIR = config["output_dir"]

def load_model(model: nn.Module, ckpt: str, device: torch.device) -> nn.Module:
    """
    Load a model from a checkpoint file.

    Args:
        model (nn.Module): The model instance to load the weights into.
        ckpt (str): Path to the checkpoint file.
        device (torch.device): The device to load the model on.

    Returns:
        nn.Module: The model loaded with the checkpoint weights.
    """
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint file {ckpt} not found.")

    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    print(f"Resume model from {ckpt}")
    return model

def process_image(
    img_path: str,
    model: nn.Module,
    transform: T.Compose,
    device: torch.device,
    dir_name: str,
    results: List[Dict],
) -> bool:
    """
    Process a single image, generate predictions, and save results.

    Args:
        img_path (str): Path to the input image.
        model (nn.Module): The model used for inference.
        transform (T.Compose): Transformations to be applied to the image.
        device (torch.device): The device to perform the computation on.
        dir_name (str): Directory to save the output labels.
        results (List[Dict]): List to store the results of processing.

    Returns:
        bool: True if the image was processed successfully, False otherwise.
    """
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return False

    img_tensor = transform(img).unsqueeze(0).to(device)
    output = model(img_tensor)
    prob = torch.softmax(output, dim=1)

    confidence = prob.max(1)[0].mean().item()
    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1).mean().item()

    pred = output.max(1)[1].cpu().numpy()[0]
    pixel_entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1).cpu().numpy()[0]

    pred[pixel_entropy > ENTROPY_THRESHOLD] = 255

    colorized_preds = Image.fromarray(pred.astype("uint8"))
    label_path = os.path.join(dir_name, f"{img_name}.png")
    colorized_preds.save(label_path)

    results.append(
        {
            "image_path": img_path,
            "label_path": label_path,
            "entropy": entropy,
            "confidence": confidence,
        }
    )

    return True

def labelgenerator(
    imagefilepaths: List[str],
    model: nn.Module,
    ckpt: str,
    bucket_idx: int = 0,
    val: bool = True,
    order: str = "asc",
) -> Tuple[List[str], str]:
    """
    Generate weak labels for a set of images.

    Args:
        imagefilepaths (List[str]): List of image file paths.
        model (nn.Module): The model used for inference.
        ckpt (str): Path to the checkpoint file.
        bucket_idx (int, optional): Index of the current bucket. Defaults to 0.
        val (bool, optional): Whether the bucket is for validation. Defaults to True.
        order (str, optional): Order of processing the buckets. Defaults to "asc".

    Returns:
        Tuple[List[str], str]: List of filtered image paths and the directory of saved labels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dir_name = os.path.join(
        WEAK_LABEL_DIR, order, f"{'val_' if val else ''}bucket_{bucket_idx}"
    )
    os.makedirs(dir_name, exist_ok=True)

    results = []
    model = load_model(model, ckpt, device)

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=config["normalization"]["mean"], std=config["normalization"]["std"]
            ),
        ]
    )

    filtered_image_paths = []
    with torch.no_grad():
        model.eval()
        for img_path in tqdm(imagefilepaths):
            if process_image(img_path, model, transform, device, dir_name, results):
                filtered_image_paths.append(img_path)

    # Compute and print overall averages
    entropy_values = [item["entropy"] for item in results]
    confidence_values = [item["confidence"] for item in results]
    print(f"Overall average entropy: {np.mean(entropy_values):.4f}")
    print(f"Overall average confidence: {np.mean(confidence_values):.4f}")

    # Save results to JSON
    json_filename = f"image_label_{'val_' if val else ''}bucket_{bucket_idx}.json"
    json_path = os.path.join(dir_name, json_filename)
    with open(json_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Saved image and label paths to {json_path}")

    return filtered_image_paths, dir_name
