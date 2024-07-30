from tqdm import tqdm
import network
import utils
import os
import argparse
import numpy as np
import json

from torch.utils import data
from datasets.kitti_360 import KittiDatasetLoader
from datasets.cityscapes_v1 import CityscapesDatasetLoader
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        help='model name')
    
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['kitti_360', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed")
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation')

    parser.add_argument("--checkpoint", type=str, required=True,
                        help='path to the checkpoint file')
    parser.add_argument("--json_file", type=str, required=True,
                        help='path to the JSON file containing image and label paths')
    
    return parser

def validate(opts, model, loader, device, metrics, image_paths, label_paths):
    metrics.reset()
    results_path = 'results' + "/" + opts.dataset
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            #for j in range(len(images)):
            #    image = images[j].detach().cpu().numpy()
            #    target = targets[j]
            #    pred = preds[j]
            #    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
            #    target = loader.dataset.decode_target(target).astype(np.uint8)
            #    pred = loader.dataset.decode_target(pred).astype(np.uint8)

            #    image_name = os.path.splitext(os.path.basename(image_paths[i * opts.val_batch_size + j]))[0]
            #    label_name = os.path.splitext(os.path.basename(label_paths[i * opts.val_batch_size + j]))[0]
            
            #    Image.fromarray(image).save(f'{results_path}/{image_name}.png')
            #    Image.fromarray(target).save(f'{results_path}/{label_name}_target.png')
            #    Image.fromarray(pred).save(f'{results_path}/{image_name}_pred.png')


            #    fig = plt.figure()
            #    plt.imshow(image)
            #    plt.axis('off')
            #    plt.imshow(pred, alpha=0.7)
            #    ax = plt.gca()
            #    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
            #    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
            #    plt.savefig(f'{results_path}/{image_name}_overlay.png', bbox_inches='tight', pad_inches=0)
            #    plt.close()

        score = metrics.get_results()
    return score


def main():
    opts = get_argparser().parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)

    # Set up model
    model = network.modeling.__dict__[opts.model](num_classes=19, output_stride=opts.output_stride)
    print("Model: %s, Output Stride: %d" % (opts.model, opts.output_stride))

    print("setting up metrics")
    # Set up metrics
    metrics = StreamSegMetrics(19)

    if opts.checkpoint is not None and os.path.isfile(opts.checkpoint):
        checkpoint = torch.load(opts.checkpoint, map_location=torch.device('cpu'))
        print("Model restored from %s" % opts.checkpoint)
        if "model_state" not in checkpoint:
            print("Key 'model_state' not found in checkpoint")
        else:
            model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    if opts.dataset == 'cityscapes':
        dataset_loader = CityscapesDatasetLoader()
    elif opts.dataset == 'kitti_360':
        dataset_loader = KittiDatasetLoader()
    else:
        raise ValueError(f"Invalid dataset: {opts.dataset}")
    
    # Read and parse the JSON file
    with open(opts.json_file, 'r') as file:
        json_data = json.load(file)
    
    # Lists to store valid image and ground truth paths
    val_image_paths = []
    val_ground_truth_paths = []
    
    # Iterate through the JSON objects
    for item in json_data:
        if item.get('image_exists') and item.get('ground_truth_exists'):
            val_image_paths.append(item.get('image'))
            val_ground_truth_paths.append(item.get('ground_truth'))    
    val_image_paths = val_image_paths[:200]
    val_ground_truth_paths = val_ground_truth_paths[:200]
    print(f"Number of validation images: {len(val_image_paths)}")
    print(f"Number of validation ground truth images: {len(val_ground_truth_paths)}")
    val_dst = dataset_loader.get_datasets(val_image_paths, val_ground_truth_paths)

    val_loader = data.DataLoader(
        val_dst,
        batch_size=opts.val_batch_size, 
        shuffle=True, 
        num_workers=2
    )

    model.eval()
    val_score = validate(opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, 
                         image_paths=val_image_paths, label_paths=val_ground_truth_paths)
    print(metrics.to_str(val_score))

if __name__ == '__main__':
    main()
