import subprocess
import json
from datetime import datetime
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter, Configuration
from dehb import DEHB
import logging
import os
import sys
import argparse

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_configuration_space():
    cs = ConfigurationSpace()
    lr = UniformFloatHyperparameter("lr", lower=1e-6, upper=1e-1, log=True)
    total_itrs = UniformFloatHyperparameter("total_itrs", lower=20, upper=2400, log=False)
    batch_size = UniformFloatHyperparameter("batch_size", lower=4, upper=64, log=True)
    weight_decay = UniformFloatHyperparameter("weight_decay", lower=1e-6, upper=1e-1, log=True)
    crop_size = UniformIntegerHyperparameter("crop_size", lower=120, upper=370)
    cs.add_hyperparameters([lr, total_itrs, batch_size, weight_decay, crop_size])
    return cs

def objective_function(config: Configuration, fidelity: float, **kwargs):
    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    pipeline_command = [
        "python", "./src/pipeline_ordered_buckets.py",
        "--buckets_order", "rand",
        "--buckets_num", str(1),
        "--total_itrs", str(int(config["total_itrs"])),
        "--lr", str(config["lr"]),
        "--batch_size", str(int(config["batch_size"])),
        "--crop_size", str(int(config["crop_size"])),
        "--weight_decay", str(config["weight_decay"]),
        "--json_input", "ranked_2k_class.json",
        "--layer", "sl",
        "--full", "True",
        "--kd", "True",
        "--pixel", "True"
    ]

    logging.info(f"Running pipeline command: {' '.join(pipeline_command)}")
    try:
        subprocess.run(pipeline_command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Pipeline command failed: {e}")
        return {"fitness": float("-inf"), "cost": fidelity}

    # Creating unique checkpoint name
    json_input = "ranked_2k_class.json"
    layer = "sl"
    full = "True"
    kd = "True"
    pixel = "True"

    checkpoint_name = f"{json_input.split('.')[0]}_{layer.upper()}_{'full' if full=='True' else 'BBN'}"
    if kd == "True":
        checkpoint_name += "_KD"
    if pixel == "True":
        checkpoint_name += "_pixel"
    
    logging.info(f"Checkpoint name: {checkpoint_name}")

    # Run the test command
    test_command = [
        "python", "./tests/test_v5.py",
        "--model", "deeplabv3plus_resnet101",
        "--gpu_id", "0",
        "--checkpoint_dir", 'checkpoints/',
        "--json_file1", "./tests/cityscapes_val_set.json",
        "--json_file2", "./tests/kitti-360_val_set_v3.json",
        "--new_ckpt", checkpoint_name,
        "--num_test", "2200"
    ]
    logging.info(f"Running test command: {' '.join(test_command)}")
    try:
        subprocess.run(test_command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Test command failed: {e}")
        return {"fitness": float("-inf"), "cost": fidelity}

    result_path = f'checkpoints/{checkpoint_name}.json'
    if not os.path.exists(result_path):
        logging.error(f"Result file not found: {result_path}")
        return {"fitness": float("-inf"), "cost": fidelity}

    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
        city_mIoU = data[0]['checkpoints'][0]["metrics_val"]['Mean IoU']
        kitti_mIoU = data[0]['checkpoints'][1]["metrics_val"]['Mean IoU']
        average_mIoU = (city_mIoU + kitti_mIoU) / 2
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error reading mIoU from result file: {e}")
        return {"fitness": float("-inf"), "cost": fidelity}

    logging.info(f"Cityscapes mIoU: {city_mIoU}")
    logging.info(f"Kitti mIoU: {kitti_mIoU}")
    logging.info(f"Average mIoU: {average_mIoU}")

    logging.info("================================================================================\n\n")
    return {"fitness": -kitti_mIoU, "cost": fidelity}

def run_dehb_optimizer(cs):
    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
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

def main():
    # Step 1: Define the Configuration Space
    cs = setup_configuration_space()

    # Step 2: Initialize and Run DEHB
    run_dehb_optimizer(cs)

if __name__ == "__main__":
    main()
