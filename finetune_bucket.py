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


# def get_argparser():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--dataset", type=str, default='cityscapes',
#                         choices=['cityscapes'], help='Name of dataset')
#     parser.add_argument("--num_classes", type=int, default=None,
#                         help="num classes (default: None)")
    
#     parser.add_argument("--bucketidx", type=int, required=True,
#                         help="bucket index to use")
#     parser.add_argument("--buckets_order", type=str, default='asc',
#                         choices=['asc', 'desc', 'rand'],
#                         help="bucket order (asc, desc or rand) (default: asc)")
#     parser.add_argument("--buckets_num", type=int, default=6,
#                         help="number of buckets (default: 6)")
#     parser.add_argument("--overwrite_old_pred", action='store_true', default=False,
#                         help="overwrite old predictions")

#     # Deeplab Options
#     available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
#                               not (name.startswith("__") or name.startswith('_')) and callable(
#                               network.modeling.__dict__[name])
#                               )
#     parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
#                         choices=available_models, help='model name')
#     parser.add_argument("--separable_conv", action='store_true', default=False,
#                         help="apply separable conv to decoder and aspp")
#     parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

#     # Train Options
#     parser.add_argument("--test_only", action='store_true', default=False)
#     parser.add_argument("--save_val_results", action='store_true', default=False,
#                         help="save segmentation results to \"./results\"")
#     parser.add_argument("--total_itrs", type=int, default=200,
#                         help="epoch number (default: 30k)")
#     parser.add_argument("--lr", type=float, default=0.01,
#                         help="learning rate (default: 0.01)")
#     parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
#                         help="learning rate scheduler policy")
#     parser.add_argument("--step_size", type=int, default=10000)
#     parser.add_argument("--crop_val", action='store_true', default=False,
#                         help='crop validation (default: False)')
#     parser.add_argument("--batch_size", type=int, default=16,
#                         help='batch size (default: 16)')
#     parser.add_argument("--val_batch_size", type=int, default=4,
#                         help='batch size for validation (default: 4)')
#     # parser.add_argument("--crop_size", type=int, default=513)
#     parser.add_argument("--crop_size", type=int, default=189)

#     parser.add_argument("--ckpt", default=None, type=str,
#                         help="restore from checkpoint")
#     parser.add_argument("--continue_training", action='store_true', default=False)

#     parser.add_argument("--loss_type", type=str, default='cross_entropy',
#                         choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
#     parser.add_argument("--gpu_id", type=str, default='0',
#                         help="GPU ID")
#     parser.add_argument("--weight_decay", type=float, default=1e-4,
#                         help='weight decay (default: 1e-4)')
#     parser.add_argument("--random_seed", type=int, default=1,
#                         help="random seed (default: 1)")
#     parser.add_argument("--print_interval", type=int, default=10,
#                         help="print interval of loss (default: 10)")
#     parser.add_argument("--val_interval", type=int, default=100,
#                         help="epoch interval for eval (default: 100)")
#     parser.add_argument("--download", action='store_true', default=False,
#                         help="download datasets")

#     return parser


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
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

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
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
        # Clean up GPU memory
        torch.cuda.empty_cache()
    return score, ret_samples


def finetuner(opts, model, checkpoint, bucket_idx, train_image_paths, val_image_paths, train_label_dir, val_label_dir,model_name , bucket_order = "asc"):
    # opts = get_argparser().parse_args()
    # if opts.dataset.lower() == 'cityscapes':
    #     opts.num_classes = 19

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

    # DataProcessor usage
    processor = DataProcessor('results.json')

    # Dictionary to map bucket orders to their respective methods
    bucket_methods = {
        'asc': processor.asc_buckets,
        'desc': processor.desc_buckets,
        'rand': processor.random_buckets
    }

    # Fetch the appropriate method based on the bucket_order
    bucket_method = bucket_methods.get(opts.buckets_order)

    # Error handling for invalid bucket_order
    if bucket_method is None:
        bucket_method = processor.asc_buckets
        raise ValueError(f"Invalid bucket_order: {opts.buckets_order}, used ascendant order instead")
        
    
    train_buckets, val_buckets = bucket_method()
    

    # Loop through the buckets starting from the specified bucketidx
    # for bucket_idx in range(opts.bucketidx ,opts.buckets_num):
    # train_image_paths = [d['image'] for d in train_buckets[bucket_idx]]
    # val_image_paths = [d['image'] for d in val_buckets[bucket_idx]]
    print(f"[INFO] Bucket number: {bucket_idx}")
    # print("[INFO] Number of Train images: %d" % len(train_image_paths))
    # print("[INFO] Number of Val images: %d" % len(val_image_paths))
    # print("Train files: %s" % train_image_paths[:2])
    # print("Val files: %s" % val_image_paths[:2])

    # train_image_paths = [d['image'] for d in train_buckets[opts.bucketidx]]
    # val_image_paths = [d['image'] for d in val_buckets[opts.bucketidx]]

    # train_json_file = f'outputs/weaklables/KITTI-360/{bucket_order}/bucket_{bucket_idx}/image_label_bucket_{bucket_idx}.json'
    # train_json_file = f'outputs/weaklables/KITTI-360/{bucket_order}/val_bucket_{bucket_idx}/image_label_bucket_{bucket_idx}.json'



    # train_image_paths = [d['image'] for d in train_buckets[bucket_idx]]
    # val_image_paths = [d['image'] for d in val_buckets[bucket_idx]]


    # train_image_paths = train_image_paths[:len(train_image_paths)//10]
    # val_image_paths = val_image_paths[:len(val_image_paths)//10]
    print("[INFO] Number of Train images: %d" % len(train_image_paths))
    print("[INFO] Number of Val images: %d" % len(val_image_paths))

    # Generate predictions for the training set if not already done
        # DatasetLoader usage
    # opts = argparse.Namespace(crop_size=512)
    dataset_loader = DatasetLoader(opts)
    
    # train_label_dir = f'outputs/weaklabels/KITTI-360/{opts.buckets_order}/bucket_{bucket_idx}'
    # val_label_dir = f'outputs/weaklabels/KITTI-360/{opts.buckets_order}/val_bucket_{bucket_idx}'

    # for dir_path in [train_label_dir, val_label_dir]:
    #     if not os.path.exists(dir_path):
    #         os.makedirs(dir_path, exist_ok=True)
    
    # train_label_paths = []
    # val_label_paths = []
    # if not opts.overwrite_old_pred:

    #     def check_images_exist(image_paths, weak_label_dir):
    #         missing_imges = []
    #         for img_path in image_paths:
    #             img_name = img_path.split('/')[-1]
    #             lbl_path = f'{weak_label_dir}/{img_name}'
    #             if not os.path.exists(lbl_path):
    #                 missing_imges.append(img_path)
    #         return missing_imges
        
    #     train_label_paths = check_images_exist(train_image_paths, train_label_dir)
    #     val_label_paths = check_images_exist(val_image_paths, val_label_dir)
    # else:
    #     train_label_paths = train_image_paths
    #     val_label_paths = val_image_paths


    # Set up model (all models are 'constructed at network.modeling)
    # model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    # print("Model: %s, Output Stride: %d" % (opts.model, opts.output_stride))

    # if opts.separable_conv and 'plus' in opts.model:
    #     network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    print("setting up metrics")
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Freeze the backbone parameters:
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Set up optimizer
    # optimizer = torch.optim.SGD(params=[
    #     {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
    #     {'params': model.classifier.parameters(), 'lr': opts.lr},
    # ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
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
    # if bucket_idx !=0:
    #     model_ckpt = 'checkpoints/latest_bucket_%s_%s_%s_os%d.pth' % (bucket_idx-1, opts.model, "kitti", opts.output_stride)
    # else:
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

    # Predict labels for the training and validation images
    # def predict_labels(model, image_paths, weak_label_dir):
    #     with torch.no_grad():
    #         model = model.eval()
    #         for img_path in tqdm(image_paths):
    #             img = Image.open(img_path).convert('RGB')
    #             img = transform(img).unsqueeze(0).to(device)
    #             output = model(img)
    #             pred = output.max(1)[1].cpu().numpy()[0] # HW
    #             pred = Image.fromarray(pred.astype('uint8'))
    #             pred.save(f'{weak_label_dir}/{img_path.split("/")[-1]}')

    # if len(train_label_paths) > 0:
    #     predict_labels(model, train_label_paths, train_label_dir)
    # print(f"[INFO] Predicted labels for {len(train_label_dir)} training images")

    # # if len(val_label_paths) > 0:
    # #     predict_labels(model, val_label_paths, val_label_dir)
    # print(f"[INFO] Predicted labels for {len(val_label_dir)} validation images")

    # ==========   Train Loop   ==========#
                
    vis_sample_id =  None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    train_dst, val_dst = dataset_loader.get_datasets(
        train_image_paths,
        train_label_dir, 
        val_image_paths, 
        val_label_dir
        )
    
    print("[INFO] Dataset Loaded......")
    train_loader = data.DataLoader(
        train_dst,
        batch_size=opts.batch_size,
        shuffle=True, 
        num_workers=2,
        drop_last=True
        )
    val_loader = data.DataLoader(
        val_dst,
        batch_size=opts.val_batch_size, 
        shuffle=True, 
        num_workers=2
        )

    print("Dataset: %s, Train set: %d, Val set: %d" %
        (opts.dataset, len(train_dst), len(val_dst)))

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        wandb.log({"Overall Test Acc": val_score['Overall Acc'], "Mean IoU": val_score['Mean IoU']})
        return
    # return

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
            # if vis is not None:
            #     vis.vis_scalar('Loss', cur_itrs, np_loss)
            wandb.log({"Loss": np_loss})

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                    (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0
                wandb.log({"Epoch": cur_epochs, "Itrs": cur_itrs, "Loss": np_loss})

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_bucket_%s_%s_%s_%s_os%d.pth' %
                        (bucket_idx, opts.buckets_order, model_name, "kitti", opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                wandb.log({"Overall Acc": val_score['Overall Acc'], "Mean IoU": val_score['Mean IoU']})

                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_bucket_%s_%s_%s_%s_os%d.pth' %
                            (bucket_idx, opts.buckets_order, model_name, "kitti", opts.output_stride))

                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                break


