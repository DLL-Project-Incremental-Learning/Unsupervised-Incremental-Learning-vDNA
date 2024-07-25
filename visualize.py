import os
import sys
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import pandas as pd

import src.network as network
from src.metrics import StreamSegMetrics
from datasets.kitti_360 import KittiDatasetLoader
from datasets.cityscapes_v1 import CityscapesDatasetLoader

# Defining the model paths
model_name = "deeplabv3plus_resnet101"
old_model_path = 'checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth'
new_model_path = 'checkpoints/ranked_2k_class_SL_full_KD_pixel.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(checkpoint_path, model_name='deeplabv3plus_resnet101', num_classes=19, output_stride=16):
    model = network.modeling.__dict__[model_name](num_classes=num_classes, output_stride=output_stride)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    return model

def load_data(json_file, dataset_name, num_test=50):
    if dataset_name == 'cityscapes':
        dataset_loader = CityscapesDatasetLoader()
    elif dataset_name == 'kitti_360':
        dataset_loader = KittiDatasetLoader()
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    with open(json_file, 'r') as file:
        json_data = json.load(file)

    val_image_paths = [item.get('image') for item in json_data if item.get('image_exists')]
    val_ground_truth_paths = [item.get('ground_truth') for item in json_data if item.get('ground_truth_exists')]

    if len(val_image_paths) > num_test:
        sampled_indices = np.random.choice(len(val_image_paths), num_test, replace=False)
        val_image_paths = [val_image_paths[i] for i in sampled_indices]
        val_ground_truth_paths = [val_ground_truth_paths[i] for i in sampled_indices]

    val_dst = dataset_loader.get_datasets(val_image_paths, val_ground_truth_paths)
    val_loader = DataLoader(val_dst, batch_size=4, shuffle=False, num_workers=2)
    return val_loader, val_image_paths, val_ground_truth_paths

def validate(model, loader, device, metrics):
    metrics.reset()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.flatten())
    return metrics.get_results(), all_preds, all_targets

def calculate_additional_metrics(all_preds, all_targets, num_classes):
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
    return {
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': cm
    }

def plot_confusion_matrix(cm, class_names, title, normalize=True, output_dir=None):
    plt.figure(figsize=(12, 10))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}.png"))
    
    plt.show()

def plot_iou_comparison(baseline_iou, finetuned_iou, class_names, dataset_name, output_dir=None):
    plt.figure(figsize=(15, 8))
    x = np.arange(len(class_names))
    width = 0.35

    baseline_values = [baseline_iou[i] for i in range(len(class_names))]
    finetuned_values = [finetuned_iou[i] for i in range(len(class_names))]

    plt.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
    plt.bar(x + width/2, finetuned_values, width, label='Fine-tuned', alpha=0.8)

    plt.xlabel('Classes')
    plt.ylabel('IoU')
    plt.title(f'Class-wise IoU Comparison - {dataset_name}')
    plt.xticks(x, class_names, rotation=90)
    plt.legend()

    for i, v in enumerate(baseline_values):
        plt.text(i - width/2, v, f'{v:.2f}', ha='center', va='bottom', rotation=90)
    for i, v in enumerate(finetuned_values):
        plt.text(i + width/2, v, f'{v:.2f}', ha='center', va='bottom', rotation=90)

    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"iou_comparison_{dataset_name}.png"))
    
    plt.show()

def save_segmentation_maps(model, image_paths, output_dir, device, num_images=5):
    os.makedirs(output_dir, exist_ok=True)
    sampled_indices = random.sample(range(len(image_paths)), num_images)
    model.eval()
    with torch.no_grad():
        for idx in sampled_indices:
            image_path = image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            image_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1).unsqueeze(0).to(device)
            output = model(image_tensor).max(1)[1].cpu().numpy()[0]
            segmentation_map = Image.fromarray(output.astype(np.uint8))

            base_name = os.path.basename(image_path).split('.')[0]
            image.save(os.path.join(output_dir, f'{base_name}_original.png'))
            segmentation_map.save(os.path.join(output_dir, f'{base_name}_segmentation.png'))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    metrics = StreamSegMetrics(19)

    baseline_model = load_model(old_model_path)
    finetuned_model = load_model(new_model_path)

    datasets = [
        ('cityscapes', 'tests/cityscapes_val_set.json'),
        ('kitti_360', 'tests/kitti-360_val_set_v3.json')
    ]

    class_names = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
        'motorcycle', 'bicycle'
    ]

    output_dir = 'visualization_results'
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name, json_file in datasets:
        print(f"\nAnalyzing {dataset_name.upper()} dataset:")
        
        val_loader, val_image_paths, _ = load_data(json_file, dataset_name)
        
        baseline_results, baseline_preds, baseline_targets = validate(baseline_model, val_loader, device, metrics)
        finetuned_results, finetuned_preds, finetuned_targets = validate(finetuned_model, val_loader, device, metrics)
        
        baseline_additional = calculate_additional_metrics(baseline_preds, baseline_targets, 19)
        finetuned_additional = calculate_additional_metrics(finetuned_preds, finetuned_targets, 19)
        
        baseline_metrics = {**baseline_results, **baseline_additional}
        finetuned_metrics = {**finetuned_results, **finetuned_additional}
        
        print("Baseline Metrics:")
        for k, v in baseline_metrics.items():
            if k != 'Confusion Matrix':
                print(f"{k}: {v}")
        
        print("\nFine-tuned Metrics:")
        for k, v in finetuned_metrics.items():
            if k != 'Confusion Matrix':
                print(f"{k}: {v}")
        
        plot_confusion_matrix(baseline_metrics['Confusion Matrix'], class_names, 
                            title=f'Normalized Confusion Matrix - {dataset_name.upper()} (Baseline)', 
                            normalize=True, output_dir=output_dir)
        plot_confusion_matrix(finetuned_metrics['Confusion Matrix'], class_names, 
                            title=f'Normalized Confusion Matrix - {dataset_name.upper()} (Fine-tuned)', 
                            normalize=True, output_dir=output_dir)
        
        baseline_iou = [baseline_metrics['Class IoU'][i] for i in range(len(class_names))]
        finetuned_iou = [finetuned_metrics['Class IoU'][i] for i in range(len(class_names))]

        plot_iou_comparison(baseline_iou, finetuned_iou, class_names, dataset_name, output_dir=output_dir)
        
        # save_segmentation_maps(baseline_model, val_image_paths, f'segmentation_maps/{dataset_name}/baseline', device)
        # save_segmentation_maps(finetuned_model, val_image_paths, f'segmentation_maps/{dataset_name}/finetuned', device)
        
        comparison_data = {
            'Metric': ['mIoU', 'Pixel Accuracy', 'Mean Accuracy', 'FWIoU', 'Precision', 'Recall', 'F1-Score'],
            'Baseline': [
                baseline_metrics['Mean IoU'],
                baseline_metrics['Overall Acc'],
                baseline_metrics['Mean Acc'],
                baseline_metrics['FreqW Acc'],
                baseline_metrics['Precision'],
                baseline_metrics['Recall'],
                baseline_metrics['F1-Score']
            ],
            'Fine-tuned': [
                finetuned_metrics['Mean IoU'],
                finetuned_metrics['Overall Acc'],
                finetuned_metrics['Mean Acc'],
                finetuned_metrics['FreqW Acc'],
                finetuned_metrics['Precision'],
                finetuned_metrics['Recall'],
                finetuned_metrics['F1-Score']
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        print(f"\nMetrics Comparison for {dataset_name.upper()}:")
        print(df.to_string(index=False))

        df.to_csv(os.path.join(output_dir, f'{dataset_name}_metrics_comparison.csv'), index=False)


if __name__ == "__main__":
    main()