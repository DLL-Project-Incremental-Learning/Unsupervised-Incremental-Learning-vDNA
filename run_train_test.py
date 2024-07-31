import subprocess
import json
import logging
import os
import warnings

# ignore warnings
warnings.filterwarnings("ignore")
# Set up logging configuration
logging.basicConfig(level=logging.INFO)


def train_test():
    """
    Executes a training and testing pipeline, processes the results, and logs the mIoU metrics.

    This function performs the following steps:
    1. Runs a training pipeline script.
    2. Runs a test script.
    3. Reads and processes the results from a JSON file.
    4. Logs and prints the mIoU metrics for Cityscapes and Kitti datasets, along with the average mIoU.
    """

    # Run the pipeline command
    pipeline_command = [
        "python",
        "./src/pipeline_ordered_buckets.py",
        "./configs/training_pipeline.json",
    ]
    logging.info(f"Running pipeline command: {' '.join(pipeline_command)}")
    try:
        subprocess.run(pipeline_command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Pipeline command failed: {e}")

    # Run the test command
    test_command = ["python", "./tests/run_test.py", "./configs/testing_pipeline.json"]
    logging.info(f"Running test command: {' '.join(test_command)}")
    try:
        subprocess.run(test_command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Test command failed: {e}")

    # Read the results
    result_path = "checkpoints/validation_results_bn.json"
    if not os.path.exists(result_path):
        logging.error(f"Result file not found: {result_path}")

    try:
        with open(result_path, "r") as f:
            data = json.load(f)
        city_mIoU = data[2]["checkpoints"][0]["metrics_val"]["Mean IoU"]
        kitti_mIoU = data[2]["checkpoints"][1]["metrics_val"]["Mean IoU"]
        average_mIoU = (city_mIoU + kitti_mIoU) / 2
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error reading mIoU from result file: {e}")

    logging.info(f"Cityscapes mIoU: {city_mIoU}")
    logging.info(f"Kitti mIoU: {kitti_mIoU}")
    logging.info(f"Average mIoU: {average_mIoU}")

    print("Cityscapes mIoU:", city_mIoU)
    print("Kitti mIoU:", kitti_mIoU)
    print("Average mIoU:", average_mIoU)

    print(
        "================================================================================\n\n"
    )


if __name__ == "__main__":
    train_test()
