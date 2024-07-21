from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

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
from dataloaders import DataProcessor, KITTI360Dataset, DatasetLoader
# Define transforms
transform = transforms.Compose([
    # transforms.Resize((512, 1024)),
    transforms.ToTensor()
])

def finetuner(opts, model, checkpoint, bucket_idx, train_image_paths, train_label_dir, model_name , bucket_order = "asc"):

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
    ### Setup dataloader

    print("[INFO] Number of Train images: %d" % len(train_image_paths))

    dataset_loader = DatasetLoader(opts)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    print("setting up metrics")
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Freeze the backbone parameters:
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(
        params=model.classifier.parameters(),
        lr=opts.lr,
        momentum=0.9,
        weight_decay=opts.weight_decay
        )

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    if opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')

    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    model_ckpt = ""
    model_ckpt = checkpoint

    if model_ckpt is not None and os.path.isfile(model_ckpt):
        checkpoint = torch.load(model_ckpt, map_location=torch.device('cpu'))
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
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % model_ckpt)
        print("Model restored from %s" % model_ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    wandb.watch(model, log="all")


    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

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
        drop_last=True
        )

    print("Dataset: %s, Train set: %d" %
        (opts.dataset, len(train_dst)))

    interval_loss = 0
    # while True:  # cur_itrs < opts.total_itrs:
    while cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        print("Epoch %d, Itrs %d/%d" % (cur_epochs, cur_itrs, opts.total_itrs))
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            wandb.log({"Loss": np_loss})

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                    (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0
                wandb.log({"Epoch": cur_epochs, "Itrs": cur_itrs, "Loss": np_loss})
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                save_ckpt('checkpoints/latest_bucket_%s_%s_%s_%s_os%d.pth' %
                          (bucket_idx, opts.buckets_order, model_name, "kitti", opts.output_stride))
                break


