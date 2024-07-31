import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.network as network
from src.metrics import StreamSegMetrics
from utils import load_config, process_dataset, generate_pdf


def main(config_path):
    """
    Main function to run the training and evaluation pipeline.

    Args:
        config_path (str): Path to the configuration JSON file.

    This function performs the following steps:
    1. Loads the configuration from the specified JSON file.
    2. Sets up the computing device (CPU or GPU).
    3. Initializes random seeds for reproducibility.
    4. Sets up the metrics for evaluation.
    5. Iterates through different checkpoint files and processes datasets.
    6. Saves the results in a JSON file and optionally generates a PDF.
    """
    config = load_config(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_id"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    print("Setting up metrics")
    metrics = StreamSegMetrics(config["num_classes"])
    results = []
    orders = ["asc", "desc", "rand"]
    for order in orders:
        results.append({"order": order, "checkpoints": []})
    checkpoint_files = [
        f
        for f in os.listdir(config["checkpoint_dir"])
        if f.endswith(".pth") and any(keyword in f for keyword in orders)
    ]

    for checkpoint_file in checkpoint_files:
        config["checkpoint_file"] = checkpoint_file
        model = network.modeling.__dict__[config["model"]](
            num_classes=config["num_classes"], output_stride=config["output_stride"]
        )
        print(
            "Model: %s, Output Stride: %d" % (config["model"], config["output_stride"])
        )
        checkpoint_path = os.path.join(config["checkpoint_dir"], checkpoint_file)
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            print("Model restored from %s" % checkpoint_path)
            if "model_state" not in checkpoint:
                print("Key 'model_state' not found in checkpoint")
            else:
                model.load_state_dict(checkpoint["model_state"])
            model = nn.DataParallel(model)
            model.to(device)
            del checkpoint
        else:
            print(f"Checkpoint file not found: {checkpoint_path}")

        process_dataset(
            config["json_file1"], "cityscapes", config, model, device, metrics, results
        )
        process_dataset(
            config["json_file2"], "kitti_360", config, model, device, metrics, results
        )

    pdf_file = os.path.join(config["checkpoint_dir"], "validation_results_bn.pdf")
    with open(
        os.path.join(config["checkpoint_dir"], "validation_results_bn.json"), "w"
    ) as file:
        json.dump(results, file, indent=4)
    # generate_pdf(results, pdf_file)
    print(f"Results saved to {pdf_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: main.py <path_to_config.json>")
        sys.exit(1)
    main(sys.argv[1])
