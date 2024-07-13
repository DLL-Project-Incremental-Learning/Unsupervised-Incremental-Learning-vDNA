import subprocess
import json
from datetime import datetime
# from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, Configuration
from dehb import DEHB
import logging
import os

def objective_function():
    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    pipeline_command = [
        "python", "pipeline_ordered_buckets.py",
        "--buckets_order", "rand",
        "--buckets_num", str(1),
        "--total_itrs", str(int(172.4484366308074)),
        "--lr", str(0.004241502795254),
        "--batch_size", str(int(6.5003187676271486)),
        "--crop_size", str(int(296)),
        "--json_input", "kitti_360_00_filtered.json",
        "--weight_decay", str(1.3680397182e-05),
        "--datetime", dt
    ]

    logging.info(f"Running pipeline command: {' '.join(pipeline_command)}")
    try:
        subprocess.run(pipeline_command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Pipeline command failed: {e}")
        # return {"fitness": float("-inf"), "cost": fidelity}

    test_command = [
        "python", "test_v5.py",
        "--model", "deeplabv3plus_resnet101",
        "--gpu_id", "0",
        "--checkpoint_dir", f'checkpoints/{dt}/',
        "--json_file1", "cityscapes_val_set.json",
        "--json_file2", "kitti-360_val_set_v3.json",
        "--num_test", "200"
    ]

    logging.info(f"Running test command: {' '.join(test_command)}")
    try:
        subprocess.run(test_command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Test command failed: {e}")
        # return {"fitness": float("-inf"), "cost": fidelity}

    result_path = f'checkpoints/{dt}/validation_results_bn.json'
    if not os.path.exists(result_path):
        logging.error(f"Result file not found: {result_path}")
        # return {"fitness": float("-inf"), "cost": fidelity}

    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
        mIoU = data[2]['checkpoints'][0]["metrics_val"]['Mean IoU']
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error reading mIoU from result file: {e}")
        # return {"fitness": float("-inf"), "cost": fidelity}

    logging.info(f"Obtained mIoU: {mIoU}")
    print("Final mIoU: ", mIoU) 

    # delete outputs/{dt} folder
    # subprocess.run(["rm", "-rf", f"outputs/{dt}"])
    # print("Deleted outputs/{dt} folder")
    # delete checkpoints/{dt}/ .pth files in the folder
    # subprocess.run(["rm", "-rf", f"checkpoints/{dt}/*.pth"])
    # print(f"Deleted checkpoints/{dt}/ .pth files in the folder")
    print("================================================================================\n\n")
    # return {"fitness": -mIoU, "cost": fidelity}


# def main():
#     objective_function()

if __name__ == "__main__":
    objective_function()
