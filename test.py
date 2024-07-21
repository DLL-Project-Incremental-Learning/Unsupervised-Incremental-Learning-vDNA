from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import json

from torch.utils.data import DataLoader, Dataset

from torch.utils import data
from datasets import Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

from datasets.kitti_360 import DatasetLoader

import torch
import torch.nn as nn
import wandb

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data/cityscapes',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")
    
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name]))
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=available_models, help='model name')

    # Deeplab Options
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    parser.add_argument("--save_val_results", action='store_true', default=True,
                        help="save segmentation results to \"./results\"")

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    
    return parser

def validate(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)   # forward pass
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            for i in range(len(images)):
                image = images[i].detach().cpu().numpy()
                target = targets[i]
                pred = preds[i]
                image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                target = loader.dataset.decode_target(target).astype(np.uint8)
                pred = loader.dataset.decode_target(pred).astype(np.uint8)
                Image.fromarray(image).save('results/%d_image.png' % img_id)
                Image.fromarray(target).save('results/%d_target.png' % img_id)
                Image.fromarray(pred).save('results/%d_pred.png' % img_id)
                fig = plt.figure()
                plt.imshow(image)
                plt.axis('off')
                plt.imshow(pred, alpha=0.7)
                ax = plt.gca()
                ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                plt.close()
                img_id += 1

        score = metrics.get_results()
    return score


def main():
    
    ckpt = "checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth"
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    # Setup wandb
    wandb.init(project="segmentation_project", config=vars(opts))
    wandb.config.update(opts)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    print("Model: %s, Output Stride: %d" % (opts.model, opts.output_stride))

    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)

    print("setting up metrics")
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    if ckpt is not None and os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
        print("Model restored from %s" % ckpt)
        if "model_state" not in checkpoint:
            print("Key 'model_state' not found in checkpoint")
        else:
            model.load_state_dict(checkpoint["model_state"])
        # model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    wandb.watch(model, log="all")

    dataset_loader = DatasetLoader()

    json_file_path = 'kitti-360_val_set_v1.json'
    
    # Read and parse the JSON file
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)
    
    # Lists to store valid image and ground truth paths
    val_image_paths = []
    val_ground_truth_paths = []
    
    # Iterate through the JSON objects
    for item in json_data:
        if item.get('image_exists') and item.get('ground_truth_exists'):
            val_image_paths.append(item.get('image'))
            val_ground_truth_paths.append(item.get('ground_truth'))    
    
    val_dst = dataset_loader.get_datasets(val_image_paths, val_ground_truth_paths)

    val_loader = data.DataLoader(
        val_dst,
        batch_size=opts.val_batch_size, 
        shuffle=True, 
        num_workers=2
        )

    model.eval()
    val_score = validate(opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
    print(metrics.to_str(val_score))
    wandb.log({"Overall Test Acc": val_score['Overall Acc'], "Mean IoU": val_score['Mean IoU']})

if __name__ == '__main__':
    main()