import json
import random
import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import network
import utils
from datasets import Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import wandb

# Load the results.json file
with open('results.json') as f:
    data = json.load(f)

# Split data into train and validation sets
random.shuffle(data)
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

# Sort and create buckets
data_sorted = sorted(train_data, key=lambda x: x['emd'])
bucket_size = len(data_sorted) // 6
buckets = [data_sorted[i:i + bucket_size] for i in range(0, len(data_sorted), bucket_size)]

# Custom dataset for KITTI360 with weak labels

# Custom dataset for KITTI360 with weak labels
class KITTI360Dataset(Dataset):

    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])



    def __init__(self, image_dir, label_dir, transform=None):
        # self.image_paths = image_paths
        self.image_paths = glob(image_dir + '/*.png')
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        target_path = os.path.join(self.label_dir, os.path.basename(img_path))
        image = Image.open(img_path).convert("RGB")
        target = Image.open(target_path)
        # print(f"img_path: {img_path}")
        # print(f"label_dir: {os.path.join(self.label_dir, os.path.basename(img_path))}")

        if self.transform:
            # image = self.transform(image)
            # target = self.transform(target)
            # target = torch.squeeze(target, 0)
            image, target = self.transform(image, target)
        return image, target

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    # parser.add_argument("--data_root", type=str, default='./datasets/data/cityscapes',
    #                     help="path to Dataset")
    parser.add_argument(
        "--data_train_imgs",
        type=str,
        default='../VisualDNA/dataset/KITTI-360/data_2d_raw/2013_05_28_drive_0007_sync/image_00/data_rect',)
    parser.add_argument(
        "--data_train_labels", 
        type=str, 
        default='../VisualDNA/dataset/KITTI-360/data_2d_semantics/2013_05_28_drive_0007_sync_labelid')
    parser.add_argument(
        "--data_val_imgs",
        type=str,
        default='../VisualDNA/dataset/KITTI-360/data_2d_raw/2013_05_28_drive_0003_sync/image_00/data_rect',)
    parser.add_argument(
        "--data_val_labels", 
        type=str, 
        default='../VisualDNA/dataset/KITTI-360/data_2d_semantics/2013_05_28_drive_0003_sync_labelid')
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=20,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    # parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--crop_size", type=int, default=189)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=20,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    return parser


def get_dataset(opts, train_data, val_data, bucket_index):
    """ Dataset And Augmentation
    """
    train_transform = et.ExtCompose([
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    current_bucket = buckets[bucket_index]

    train_dst = KITTI360Dataset(
        image_dir=[item['image'] for item in current_bucket], 
        label_dir=opts.data_train_labels,
        transform=train_transform
        )
    val_dst = KITTI360Dataset(
        image_dir=[item['image'] for item in val_data],
        label_dir=opts.data_val_labels,
        transform=val_transform
        )
    return train_dst, val_dst

def pseudo_labeling(model, loader, device):
    model.eval()
    pseudo_labels = []
    with torch.no_grad():
        for images, _ in tqdm(loader):
            images = images.to(device, dtype=torch.float32)
            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            pseudo_labels.append(preds)
    return pseudo_labels

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
    return score, ret_samples


def main():
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

    # Load model
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        print("Model restored from %s" % opts.ckpt)
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
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    wandb.watch(model, log="all")

    
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    model = nn.DataParallel(model)
    model.to(device)

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

    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    metrics = StreamSegMetrics(opts.num_classes)

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
    # Freeze the backbone parameters
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Start incremental self-training
    for bucket_index in range(6):
        print(f"Training with bucket {bucket_index + 1} of 6")

        train_dst, val_dst = get_dataset(opts, train_data, val_data, bucket_index)
        train_loader = DataLoader(
            train_dst,
            batch_size=opts.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dst,
            batch_size=opts.val_batch_size,
            shuffle=True,
            num_workers=2
        )

        # Weak labeling for the current bucket
        pseudo_labels = pseudo_labeling(model, train_loader, device)
        for i, (image, label) in enumerate(train_dst):
            label = pseudo_labels[i]

        print(f"Bucket {bucket_index + 1} - pseudo labels generated")

        # Train the model with pseudo-labeled data
        cur_itrs = 0
        cur_epochs = 0
        best_score = 0.0
        while cur_itrs < opts.total_itrs:
            model.train()
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

                interval_loss += loss.detach().cpu().numpy()
                if (cur_itrs) % 10 == 0:
                    interval_loss = interval_loss / 10
                    print("Epoch %d, Itrs %d/%d, Loss=%f" %
                          (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                    interval_loss = 0.0

                if (cur_itrs) % opts.val_interval == 0:
                    save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))
                    print("validation...")
                    model.eval()
                    val_score, ret_samples = validate(
                        opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                        ret_samples_ids=vis_sample_id)
                    print(metrics.to_str(val_score))

                    if val_score['Mean IoU'] > best_score:  # save best model
                        best_score = val_score['Mean IoU']
                        save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                                  (opts.model, opts.dataset, opts.output_stride))

                    model.train()
                scheduler.step()

                if cur_itrs >= opts.total_itrs:
                    break
            if cur_itrs >= opts.total_itrs:
                break

if __name__ == '__main__':
    main()
