from tqdm import tqdm
import network
import utils
import os
import sys
import random
import argparse
import numpy as np

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils import data
from datasets import Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
import wandb

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from torchvision import transforms

# from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader, Dataset
from glob import glob
from collections import namedtuple
from datasets.dataloaders import DataProcessor, KITTI360Dataset, DatasetLoader

import torch.nn.functional as F

# Define transforms
transform = transforms.Compose(
    [
        # transforms.Resize((512, 1024)),
        transforms.ToTensor()
    ]
)


# Define the KnowledgeDistillationLoss class
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction="mean", alpha=1.0):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])
        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)
        loss = (outputs * labels).mean(dim=1)

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == "mean":
            outputs = -torch.mean(loss)
        elif self.reduction == "sum":
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


def gradual_unfreezing(model, current_itrs, total_itrs):
    # Define the unfreezing schedule
    if isinstance(model, torch.nn.DataParallel):
        # Access the original model to get the backbone
        backbone_model = model.module.backbone
    else:
        # Directly access the backbone if not wrapped in DataParallel
        backbone_model = model.backbone

    unfreeze_schedule = {
        "layer4": int(total_itrs * 0.2),  # Unfreeze at 20% of total iterations
        "layer3": int(total_itrs * 0.4),  # Unfreeze at 40% of total iterations
        "layer2": int(total_itrs * 0.6),  # Unfreeze at 60% of total iterations
        "layer1": int(total_itrs * 0.8),  # Unfreeze at 80% of total iterations
    }

    # Unfreeze layers based on the current iteration
    for layer_name, unfreeze_at in unfreeze_schedule.items():
        if current_itrs >= unfreeze_at:
            layer = getattr(backbone_model, layer_name)
            for param in layer.parameters():
                param.requires_grad = True
            print(f"[INFO] Unfreezing {layer_name} at iteration {current_itrs}")


def finetuner(
    opts,
    model,
    teacher_model,
    teacher_ckpt,
    checkpoint,
    bucket_idx,
    train_image_paths,
    train_label_dir,
    model_name,
    bucket_order="asc",
):

    # Setup wandb
    # wandb.init(project="segmentation_project", config=vars(opts))
    # wandb.config.update(opts)

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    # Creating unique checkpoint name
    json_input = opts.json_input
    layer = opts.layer
    full = opts.full
    kd = opts.kd
    pixel = opts.pixel

    checkpoint_name = f"{json_input.split('.')[0]}_{layer.upper()}_{'full' if full=='True' else 'BBN'}"
    if kd == "True":
        checkpoint_name += "_KD"
    if pixel== "True":
        checkpoint_name += "_pixel"


    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    print("[INFO] Number of Train images: %d" % len(train_image_paths))

    dataset_loader = DatasetLoader(opts)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    print("setting up metrics")
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    def print_model_structure(model):
        for name, module in model.named_children():
            print(f"Layer: {name}")
            if hasattr(module, "named_children"):
                for sub_name, sub_module in module.named_children():
                    print(f"  Sub-layer: {sub_name}")

    print_model_structure(model)

    # Freeze the backbone parameters:
    for param in model.backbone.parameters():
        param.requires_grad = False


    if opts.layer =="sl":
        layers_to_unfreeze = ['layer1.0', 'layer1.1', 'layer1.2', 'layer2.0', 'layer3.1.', 'layer3.2.']
        for name, param in model.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True

    elif opts.layer =="gl":
        print("Gradual Unfreezing")

    elif int(opts.layer[1]) < 5:
        layer_num = opts.layer[1]
        # Unfreeze the last layer of the backbone 
        for name, param in model.named_parameters():
            if f"layer{layer_num}." in name:
                param.requires_grad = True

    print("----Opts FULL:-------", opts.full)
    # freeze the segmentation head
    if opts.full == "True":
        for param in model.classifier.parameters():
            param.requires_grad = True
            
    elif opts.full == "False":
        print("Inside BBN")
        for param in model.classifier.parameters():
            param.requires_grad = False
            
    # Unfreeze only the bias terms in the classifier
        for name, param in model.classifier.named_parameters():
            if 'bias' in name or 'bn' in name:
                param.requires_grad = True

    # Optional: Print the trainable parameters to verify
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters: {count_parameters(model)}")

    # Optional: Print which parts of the model are trainable
    def print_trainable_parameters(model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}")
            else:
                print(f"Frozen: {name}")

    print_trainable_parameters(model)

    # Set up optimizer
    optimizer = torch.optim.SGD(
        params=[
            {"params": model.backbone.parameters(), "lr": 0.1 * opts.lr},
            {"params": model.classifier.parameters()},
        ],
        lr=opts.lr,
        momentum=0.9,
        weight_decay=opts.weight_decay,
    )

    if opts.lr_policy == "poly":
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opts.step_size, gamma=0.1
        )

    # Set up criterion
    if opts.loss_type == "cross_entropy":
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="none")
    elif opts.loss_type == "focal_loss":
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)

    def save_ckpt(path):
        """save current model"""
        torch.save(
            {
                "cur_itrs": cur_itrs,
                "model_state": model.module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_score": best_score,
            },
            path,
        )
        print("Model saved as %s" % path)

    utils.mkdir("./checkpoints")

    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    model_ckpt = checkpoint

    if model_ckpt is not None and os.path.isfile(model_ckpt):
        checkpoint = torch.load(model_ckpt, map_location=torch.device("cpu"))
        print("Model restored from %s" % model_ckpt)
        if "model_state" not in checkpoint:
            print("Key 'model_state' not found in checkpoint")
        else:
            model.load_state_dict(checkpoint["model_state"])
        # model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint["best_score"]
            print("Training state restored from %s" % model_ckpt)
        print("Model restored from %s" % model_ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # wandb.watch(model, log="all")

    teacher_model_ckpt = teacher_ckpt
    teacher_model.load_state_dict(
        torch.load(teacher_model_ckpt, map_location=torch.device("cpu"))["model_state"]
    )
    teacher_model = nn.DataParallel(teacher_model)
    teacher_model.to(device)
    teacher_model.eval()

    train_dst = dataset_loader.get_datasets(
        train_image_paths,
        train_label_dir,
    )

    print("[INFO] Dataset Loaded......")
    train_loader = data.DataLoader(
        train_dst,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    print("Dataset: %s, Train set: %d" % (opts.dataset, len(train_dst)))

    kd_loss_fn = KnowledgeDistillationLoss(reduction="mean", alpha=1.0)
    interval_loss = 0
    # while True:  # cur_itrs < opts.total_itrs:
    while cur_itrs < opts.total_itrs:

        if opts.layer == "gl":        
            gradual_unfreezing(model, cur_itrs, opts.total_itrs)
        # =====  Train  =====
        model.train()
        print("Epoch %d, Itrs %d/%d" % (cur_epochs, cur_itrs, opts.total_itrs))
        cur_epochs += 1
        for images, labels in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)

            # Get teacher predictions
            with torch.no_grad():
                teacher_outputs = teacher_model(images)

            if opts.kd == "True":
                # print("Using Knowledge Distillation Loss------------------------------------")
                distillation_loss = kd_loss_fn(outputs, teacher_outputs)
                segmentation_loss = criterion(outputs, labels).mean()  # scalar
                loss = 1 * distillation_loss + segmentation_loss
            else:
                # print("Using Segmentation Loss------------------------------------")
                loss = criterion(outputs, labels).mean() 

            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            # wandb.log({"Loss": np_loss})

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print(
                    "Epoch %d, Itrs %d/%d, Loss=%f"
                    % (cur_epochs, cur_itrs, opts.total_itrs, interval_loss)
                )
                interval_loss = 0.0
                # wandb.log({"Epoch": cur_epochs, "Itrs": cur_itrs, "Loss": np_loss})
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                save_ckpt(
                    f"./checkpoints/{checkpoint_name}.pth"    
                )
                break
