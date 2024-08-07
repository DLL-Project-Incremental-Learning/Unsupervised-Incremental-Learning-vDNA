import json
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from datasets.kitti_360 import KittiDatasetLoader
from datasets.cityscapes_v1 import CityscapesDatasetLoader


def validate(opts, model, loader, device, metrics, image_paths, label_paths):
    """
    Validate the model on the given data loader.

    Args:
        opts (dict): Configuration options.
        model (torch.nn.Module): Model to validate.
        loader (torch.utils.data.DataLoader): Data loader for validation data.
        device (torch.device): Device to run the validation on (CPU or GPU).
        metrics (StreamSegMetrics): Metrics object to update and retrieve results.
        image_paths (list): List of image file paths.
        label_paths (list): List of label file paths.

    Returns:
        dict: Validation scores.
    """
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
    """
    Generate a PDF report of the validation results.

    Args:
        results (list): List of results dictionaries.
        output_file (str): Path to save the generated PDF.
    """
    doc = SimpleDocTemplate(output_file, pagesize=landscape(letter))
    elements = []

    data = [
        [
            "Order",
            "Dataset",
            "Checkpoint Name",
            "Overall Accuracy",
            "Mean Accuracy",
            "FreqW Accu",
            "Mean mIoU",
        ]
    ]
    miou_values = []

    for result in results:
        order = result["order"]
        for checkpoint in result["checkpoints"]:
            metrics_val = checkpoint["metrics_val"]
            miou = metrics_val["Mean IoU"]
            miou_values.append(f"{miou:.6f}")
            data.append(
                [
                    order,
                    checkpoint["dataset"],
                    checkpoint["name"],
                    f"{metrics_val['Overall Acc']:.6f}",
                    f"{metrics_val['Mean Acc']:.6f}",
                    f"{metrics_val['FreqW Acc']:.6f}",
                    f"{miou:.6f}",
                ]
            )

    # Find the min and max mIoU values
    min_miou = float(min(miou_values))
    max_miou = float(max(miou_values))

    table = Table(data)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.white),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Roman"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )

    # Apply color to min and max mIoU values
    for i in range(1, len(data)):
        miou = float(data[i][6])
        if miou == min_miou:
            table.setStyle(TableStyle([("BACKGROUND", (6, i), (6, i), colors.red)]))
        elif miou == max_miou:
            table.setStyle(TableStyle([("BACKGROUND", (6, i), (6, i), colors.green)]))

    elements.append(table)
    doc.build(elements)


def process_dataset(json_file, dataset_name, config, model, device, metrics, results):
    """
    Process a dataset for validation.

    Args:
        json_file (str): Path to the JSON file containing dataset information.
        dataset_name (str): Name of the dataset ('cityscapes' or 'kitti_360').
        config (dict): Configuration options.
        model (torch.nn.Module): Model to validate.
        device (torch.device): Device to run the validation on (CPU or GPU).
        metrics (StreamSegMetrics): Metrics object to update and retrieve results.
        results (list): List to store the validation results.
    """
    # print(json_file)
    if dataset_name == "cityscapes":
        dataset_loader = CityscapesDatasetLoader()
    elif dataset_name == "kitti_360":
        dataset_loader = KittiDatasetLoader()
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    with open(json_file, "r") as file:
        json_data = json.load(file)

    val_image_paths = []
    val_ground_truth_paths = []
    for item in json_data:
        if item.get("image_exists") and item.get("ground_truth_exists"):
            val_image_paths.append(item.get("image"))
            val_ground_truth_paths.append(item.get("ground_truth"))

    # Randomly sample 500 images
    if len(val_image_paths) > config["num_test"]:
        sampled_indices = np.random.choice(
            len(val_image_paths), config["num_test"], replace=False
        )
        val_image_paths = [val_image_paths[i] for i in sampled_indices]
        val_ground_truth_paths = [val_ground_truth_paths[i] for i in sampled_indices]

    print(f"Number of validation images for {dataset_name}: {len(val_image_paths)}")
    print(
        f"Number of validation ground truth images for {dataset_name}: {len(val_ground_truth_paths)}"
    )

    val_dst = dataset_loader.get_datasets(val_image_paths, val_ground_truth_paths)
    val_loader = data.DataLoader(
        val_dst, batch_size=config["val_batch_size"], shuffle=True, num_workers=2
    )

    model.eval()
    val_score = validate(
        opts=config,
        model=model,
        loader=val_loader,
        device=device,
        metrics=metrics,
        image_paths=val_image_paths,
        label_paths=val_ground_truth_paths,
    )
    print(metrics.to_str(val_score))

    class_names = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]

    # metrics.plot_confusion_matrix(class_names = class_names)

    for result in results:
        if result["order"] in config["checkpoint_file"]:
            result["checkpoints"].append(
                {
                    "name": config["checkpoint_file"],
                    "metrics_val": val_score,
                    "dataset": dataset_name,
                }
            )
            break


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)
