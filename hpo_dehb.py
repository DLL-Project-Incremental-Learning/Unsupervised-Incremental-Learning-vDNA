from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter
import subprocess
import json
from datetime import datetime
from ConfigSpace import Configuration
from dehb import DEHB
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Define the Configuration Space
cs = ConfigurationSpace()

lr = UniformFloatHyperparameter("lr", lower=1e-6, upper=1e-1, log=True)
total_itrs = UniformIntegerHyperparameter("total_itrs", lower=1, upper=100)
batch_size = UniformIntegerHyperparameter("batch_size", lower=1, upper=64)
weight_decay = UniformFloatHyperparameter("weight_decay", lower=1e-6, upper=1e-1, log=True)

cs.add_hyperparameters([lr, total_itrs, batch_size, weight_decay])

# Step 2: Adapt the Objective Function
def objective_function(config: Configuration, fidelity: float, **kwargs):
    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    pipeline_command = [
        "python", "pipeline_ordered_buckets.py",
        "--buckets_order", "rand",
        "--buckets_num", str(1),
        "--total_itrs", str(config["total_itrs"]),
        "--lr", str(config["lr"]),
        "--batch_size", str(config["batch_size"]),
        "--crop_size", str(370),
        "--json_input", "kitti_360_00_filtered.json",
        "--weight_decay", str(config["weight_decay"]),
        "--datetime", dt
    ]

    logging.info(f"Running pipeline command: {' '.join(pipeline_command)}")
    try:
        subprocess.run(pipeline_command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Pipeline command failed: {e}")
        return {"fitness": float("-inf"), "cost": fidelity}

    test_command = [
        "python", "test_v5.py",
        "--model", "deeplabv3plus_resnet101",
        "--gpu_id", "0",
        "--checkpoint_dir", f'checkpoints/{dt}/',
        "--json_file1", "cityscapes_val_set.json",
        "--json_file2", "kitti-360_val_set_v3.json",
        "--num_test", "20"
    ]

    logging.info(f"Running test command: {' '.join(test_command)}")
    try:
        subprocess.run(test_command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Test command failed: {e}")
        return {"fitness": float("-inf"), "cost": fidelity}

    result_path = f'checkpoints/{dt}/validation_results_bn.json'
    if not os.path.exists(result_path):
        logging.error(f"Result file not found: {result_path}")
        return {"fitness": float("-inf"), "cost": fidelity}

    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
        mIoU = data[2]['checkpoints'][0]["metrics_val"]['Mean IoU']
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error reading mIoU from result file: {e}")
        return {"fitness": float("-inf"), "cost": fidelity}

    logging.info(f"Obtained mIoU: {mIoU}")
    return {"fitness": mIoU, "cost": fidelity}

# Step 3: Initialize and Run DEHB
dim = len(cs.get_hyperparameters())
optimizer = DEHB(
    f=objective_function,
    cs=cs,
    dimensions=dim,
    min_fidelity=1,
    max_fidelity=100,
    eta=3,
    n_workers=1,
    output_path="./logs",
)

# Run optimization for 1 bracket. Output files will be saved to ./logs
traj, runtime, history = optimizer.run(brackets=1)

# Extract and print the results
for entry in history:
    config_id, config, fitness, runtime, fidelity, _ = entry
    logging.info(f"Config ID: {config_id}")
    logging.info(f"Config: {config}")
    logging.info(f"Fitness: {fitness}")
    logging.info(f"Runtime: {runtime}")
    logging.info(f"Fidelity: {fidelity}")
