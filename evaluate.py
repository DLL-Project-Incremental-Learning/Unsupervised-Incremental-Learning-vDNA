import subprocess
import json
import logging
import os

logging.basicConfig(level=logging.INFO)

def objective_function(hyperparameters_file):
    # Load hyperparameters from the JSON file
    try:
        with open(hyperparameters_file, 'r') as f:
            data = json.load(f)
        hyperparameters = data["config"]
        logging.info(f"Loaded hyperparameters: {hyperparameters}")
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        logging.error(f"Error loading hyperparameters: {e}")
        return {"fitness": float("-inf"), "cost": 1}
    
    # Run the pipeline command
    pipeline_command = [
        "python", "pipeline_ordered_buckets.py",
        "--buckets_order", "rand",
        "--buckets_num", str(1),
        "--total_itrs", str(int(hyperparameters["total_itrs"])),
        "--lr", str(hyperparameters["lr"]),
        "--batch_size", str(int(hyperparameters["batch_size"])),
        "--crop_size", str(int(hyperparameters["crop_size"])),
        "--json_input", "kitti_360_00_filtered.json",
        "--weight_decay", str(hyperparameters["weight_decay"])
    ]
    logging.info(f"Running pipeline command: {' '.join(pipeline_command)}")
    try:
        subprocess.run(pipeline_command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Pipeline command failed: {e}")
        return {"fitness": float("-inf"), "cost": 1}
    
    # Run the test command
    test_command = [
        "python", "test_v5.py",
        "--model", "deeplabv3plus_resnet101",
        "--gpu_id", "0",
        "--checkpoint_dir", 'checkpoints/',
        "--json_file1", "cityscapes_val_set.json",
        "--json_file2", "kitti-360_val_set_v3.json",
        "--num_test", "200"
    ]
    logging.info(f"Running test command: {' '.join(test_command)}")
    try:
        subprocess.run(test_command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Test command failed: {e}")
        return {"fitness": float("-inf"), "cost": 1}
    
    # Read the results
    result_path = 'checkpoints/validation_results_bn.json'
    if not os.path.exists(result_path):
        logging.error(f"Result file not found: {result_path}")
        return {"fitness": float("-inf"), "cost": 1}

    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
        mIoU = data[2]['checkpoints'][0]["metrics_val"]['Mean IoU']
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error reading mIoU from result file: {e}")
        return {"fitness": float("-inf"), "cost": 1}

    logging.info(f"Obtained mIoU: {mIoU}")
    print("Final mIoU: ", mIoU)
    
    # Cleanup if needed
    # Uncomment these lines if cleanup is needed
    # subprocess.run(["rm", "-rf", "outputs/"])
    # logging.info(f"Deleted outputs folder")
    # subprocess.run(["rm", "-rf", "checkpoints/*.pth"])
    # logging.info(f"Deleted checkpoints .pth files in the folder")

    print("================================================================================\n\n")
    return {"fitness": -mIoU, "cost": 1}

if __name__ == "__main__":
    hyperparameters_file = "logs/incumbent.json"
    objective_function(hyperparameters_file)