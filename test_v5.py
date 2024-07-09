from tqdm import tqdm
import network
import utils
import os
import argparse
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils import data
from datasets.kitti_360 import KittiDatasetLoader
from datasets.cityscapes_v1 import CityscapesDatasetLoader
from metrics import StreamSegMetrics
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101', help='model name')
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument("--val_batch_size", type=int, default=4, help='batch size for validation')
    parser.add_argument("--checkpoint_dir", default='checkpoints\\', type=str, required=True, help='path to the directory containing checkpoint files')
    parser.add_argument("--json_file1", default='cityscapes_val_set.json', type=str, required=True, help='path to the first JSON file containing image and label paths')
    parser.add_argument("--json_file2", default='kitti-360_val_set_v1.json', type=str, required=True, help='path to the second JSON file containing image and label paths')
    parser.add_argument("--num_test", type=int, default=250, help='number of test images')
    return parser

def validate(opts, model, loader, device, metrics, image_paths, label_paths):
    metrics.reset()
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)
        score = metrics.get_results()
    return score

def generate_pdf(results, output_file):
    doc = SimpleDocTemplate(output_file, pagesize=landscape(letter))
    elements = []

    data = [["Order", "Dataset", "Checkpoint Name", "Overall Accuracy", "Mean Accuracy", "FreqW Accu", "Mean mIoU"]]
    miou_values = []

    for result in results:
        print(result)
        order = result['order']
        for checkpoint in result['checkpoints']:
            metrics_val = checkpoint['metrics_val']
            miou = metrics_val['Mean IoU']
            miou_values.append(f"{miou:.6f}")
            data.append([
                order,
                checkpoint['dataset'],
                checkpoint['name'],
                f"{metrics_val['Overall Acc']:.6f}",
                f"{metrics_val['Mean Acc']:.6f}",
                f"{metrics_val['FreqW Acc']:.6f}",
                f"{miou:.6f}"
            ])

    # Find the min and max mIoU values
    min_miou = float(min(miou_values))
    max_miou = float(max(miou_values))

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Roman'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    # Apply color to min and max mIoU values
    for i in range(1, len(data)):
        miou = float(data[i][6])
        if miou == min_miou:
            table.setStyle(TableStyle([('BACKGROUND', (6, i), (6, i), colors.red)]))
        elif miou == max_miou:
            table.setStyle(TableStyle([('BACKGROUND', (6, i), (6, i), colors.green)]))

    elements.append(table)
    doc.build(elements)

def process_dataset(json_file, dataset_name, opts, model, device, metrics, results):
    if dataset_name == 'cityscapes':
        dataset_loader = CityscapesDatasetLoader()
    elif dataset_name == 'kitti_360':
        dataset_loader = KittiDatasetLoader()
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    with open(json_file, 'r') as file:
        json_data = json.load(file)
    
    val_image_paths = []
    val_ground_truth_paths = []
    for item in json_data:
        if item.get('image_exists') and item.get('ground_truth_exists'):
            val_image_paths.append(item.get('image'))
            val_ground_truth_paths.append(item.get('ground_truth'))

    # Randomly sample 500 images
    if len(val_image_paths) > opts.num_test:
        sampled_indices = np.random.choice(len(val_image_paths), opts.num_test, replace=False)
        val_image_paths = [val_image_paths[i] for i in sampled_indices]
        val_ground_truth_paths = [val_ground_truth_paths[i] for i in sampled_indices]

    print(f"Number of validation images for {dataset_name}: {len(val_image_paths)}")
    print(f"Number of validation ground truth images for {dataset_name}: {len(val_ground_truth_paths)}")
    
    val_dst = dataset_loader.get_datasets(val_image_paths, val_ground_truth_paths)
    val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    
    model.eval()
    val_score = validate(opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, image_paths=val_image_paths, label_paths=val_ground_truth_paths)
    print(metrics.to_str(val_score))

    class_names = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 
    'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 
    'motorcycle', 'bicycle']

    # metrics.plot_confusion_matrix(class_names = class_names)
    
    for result in results:
        if result['order'] in opts.checkpoint_file:
            result['checkpoints'].append({
                "name": opts.checkpoint_file,
                "metrics_val": val_score,
                "dataset": dataset_name
            })
            break

def main():
    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    print("setting up metrics")
    metrics = StreamSegMetrics(19)
    results = []
    orders = ['asc', 'desc', 'rand']
    for order in orders:
        results.append({'order': order, 'checkpoints': []})
    checkpoint_files = [f for f in os.listdir(opts.checkpoint_dir) if f.endswith('.pth') and any(keyword in f for keyword in orders)]
    
    for checkpoint_file in checkpoint_files:
        opts.checkpoint_file = checkpoint_file
        model = network.modeling.__dict__[opts.model](num_classes=19, output_stride=opts.output_stride)
        print("Model: %s, Output Stride: %d" % (opts.model, opts.output_stride))
        checkpoint_path = os.path.join(opts.checkpoint_dir, checkpoint_file)
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
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
        
        # process_dataset(opts.json_file1, 'cityscapes', opts, model, device, metrics, results)
        process_dataset(opts.json_file2, 'kitti_360', opts, model, device, metrics, results)

    pdf_file = os.path.join(opts.checkpoint_dir, 'validation_results_bn.pdf')
    generate_pdf(results, pdf_file)
    print(f"Results saved to {pdf_file}")

if __name__ == '__main__':
    main()
