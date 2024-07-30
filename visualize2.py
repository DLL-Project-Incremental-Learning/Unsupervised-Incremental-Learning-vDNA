import torch
import torch.nn as nn
from torch.utils import data
import json
import numpy as np
import random
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
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

# Load model
def load_model(checkpoint_path, model_name='deeplabv3plus_resnet101', num_classes=19, output_stride=16):
    model = network.modeling.__dict__[model_name](num_classes=num_classes, output_stride=output_stride)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    return model

# Load data
def load_data(json_file, dataset_name, num_test=400):
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
        print(f"Sampled {num_test} images from the dataset")

    val_dst = dataset_loader.get_datasets(val_image_paths, val_ground_truth_paths)
    val_loader = data.DataLoader(val_dst, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    return val_loader, val_image_paths, val_ground_truth_paths

# Validate model
def validate(model, loader, device, metrics):
    metrics.reset()
    scaler = GradScaler()

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            with autocast():
                outputs = model(images)
            
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)

            # Clear cache to free memory
            del images, labels, outputs
            torch.cuda.empty_cache()
    
    return metrics.get_results()

# Calculate additional metrics
def calculate_additional_metrics(metrics, num_classes):
    precision, recall, f1, _ = precision_recall_fscore_support(metrics.all_targets, metrics.all_preds, average='weighted')
    cm = confusion_matrix(metrics.all_targets, metrics.all_preds, labels=range(num_classes))
    return {
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': cm
    }

def plot_confusion_matrix(cm, class_names, title=None, normalize=True):
    plt.figure(figsize=(12, 10))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    if title:
        plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
metrics = StreamSegMetrics(19)

# Load models
baseline_model = load_model(old_model_path)
finetuned_model = load_model(new_model_path)

# Load data
datasets = [
    ('cityscapes', 'tests/cityscapes_val_set.json'),
    ('kitti_360', 'tests/kitti-360_val_set_v3.json')
]

class_names = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
    'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle'
]

for dataset_name, json_file in datasets:
    print(f"\nAnalyzing {dataset_name.upper()} dataset:")
    
    # Load data
    val_loader, val_image_paths, val_ground_truth_paths = load_data(json_file, dataset_name, num_test=2000)
    
    finetuned_results = validate(finetuned_model, val_loader, device, metrics)
    
    fig = metrics.plot_confusion_matrix()
    # Display the plot in Jupyter
    display(fig)
    # Save the figure
    fig.savefig(f'confusion_matrix_Finetuned_{dataset_name}.png', bbox_inches='tight', dpi=300)
    # Close the figure to free up memory
    plt.close(fig)
    
    finetuned_additional = calculate_additional_metrics(metrics, num_classes=19)
    
    # Print metrics
    print("\nFine-tuned Metrics:")
    for k, v in {**finetuned_results, **finetuned_additional}.items():
        if k != 'Confusion Matrix':
            print(f"{k}: {v}")

# Define function to plot IoU comparison
def plot_iou_comparison(baseline_iou, finetuned_iou, class_names, dataset_name):
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

    # Add value labels on top of each bar
    for i, v in enumerate(baseline_values):
        plt.text(i - width/2, v, f'{v:.2f}', ha='center', va='bottom', rotation=90)
    for i, v in enumerate(finetuned_values):
        plt.text(i + width/2, v, f'{v:.2f}', ha='center', va='bottom', rotation=90)

    plt.tight_layout()
    plt.show()

# Save segmentation maps
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

# Plot IoU comparison
baseline_iou = [baseline_metrics['Class IoU'][i] for i in range(len(class_names))]
finetuned_iou = [finetuned_metrics['Class IoU'][i] for i in range(len(class_names))]

plot_iou_comparison(baseline_iou, finetuned_iou, class_names, dataset_name)

# Create a comparison table
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
