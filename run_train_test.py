import subprocess
import json
import logging
import os
import argparse
logging.basicConfig(level=logging.INFO)

def objective_function(args):

    # Run the pipeline command
    pipeline_command = [
        "python", "./src/pipeline_ordered_buckets.py",
        "--buckets_order", "rand",
        "--buckets_num", str(1),
        "--total_itrs", str(10),
        "--lr", str(0.014),
        "--batch_size", str(4),
        "--crop_size", str(370),
        "--weight_decay", str(3e-5),
        "--json_input", args.json_input,
        "--layer", args.layer,
        "--full", str(args.full),
        "--kd", str(args.kd),
        "--pixel", str(args.pixel)
    ]

    logging.info(f"Running pipeline command: {' '.join(pipeline_command)}")
    try:
        subprocess.run(pipeline_command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Pipeline command failed: {e}")
        return {"fitness": float("-inf"), "cost": 1}
    
    # Creating unique checkpoint name
    json_input = args.json_input
    layer = args.layer
    full = args.full
    kd = args.kd
    pixel = args.pixel

    checkpoint_name = f"{json_input.split('.')[0]}_{layer.upper()}_{'full' if full=='True' else 'BBN'}"
    if kd == "True":
        checkpoint_name += "_KD"
    if pixel == "True":
        checkpoint_name += "_pixel"
    
    print("Checkpoint name:", checkpoint_name)

    # Run the test command
    test_command = [
        "python", "./tests/test_v5.py",
        "--model", "deeplabv3plus_resnet101",
        "--gpu_id", "0",
        "--checkpoint_dir", 'checkpoints/',
        "--json_file1", "./tests/cityscapes_val_set.json",
        "--json_file2", "./tests/kitti-360_val_set_v3.json",
        "--new_ckpt",checkpoint_name, 
        "--num_test", "25"
    ]
    logging.info(f"Running test command: {' '.join(test_command)}")
    try:
        subprocess.run(test_command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Test command failed: {e}")
        return {"fitness": float("-inf"), "cost": 1}
    
    # Read the results
    result_path = f'checkpoints/{checkpoint_name}.json'
    if not os.path.exists(result_path):
        logging.error(f"Result file not found: {result_path}")
        return {"fitness": float("-inf"), "cost": 1}

    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
        city_mIoU = data[0]['checkpoints'][0]["metrics_val"]['Mean IoU']
        kitti_mIoU = data[0]['checkpoints'][1]["metrics_val"]['Mean IoU']
        average_mIoU = (city_mIoU + kitti_mIoU) / 2
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error reading mIoU from result file: {e}")
        return {"fitness": float("-inf"), "cost": 1}

    # logging.info(f"Cityscapes mIoU: {city_mIoU}")
    # logging.info(f"Kitti mIoU: {kitti_mIoU}")
    # logging.info(f"Average mIoU: {average_mIoU}")

    print("Cityscapes mIoU:", city_mIoU)
    print("Kitti mIoU:", kitti_mIoU)
    print("Average mIoU:", average_mIoU)


    print("================================================================================\n\n")
    return {"fitness": -average_mIoU, "cost": 1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_input", type=str, default="rank_1_val.json")
    parser.add_argument("--layer", type=str, default="l1", help="layer number",
        choices=["l1","l2","l3","l4","l5","sl","gl"])
    parser.add_argument("--full", type=str, default="True", help="full or Bias BN",
        choices=["True","False"])
    parser.add_argument("--kd", type=str, default="False", help="Knowledge distillation")
    parser.add_argument("--pixel", type=str, default="False", help="Pixel level distillation")
    args = parser.parse_args()
    objective_function(args)

if __name__ == "__main__":
    main()

    