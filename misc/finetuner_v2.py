import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
import wandb
from tqdm import tqdm
import copy

# Assume these are imported from your existing files
from network import DeeplabV3Plus
from utils import set_bn_momentum, PolyLR, FocalLoss, Denormalize
from metrics import StreamSegMetrics
from dataloaders import DatasetLoader

class Trainer:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup random seed
        torch.manual_seed(opts.random_seed)
        np.random.seed(opts.random_seed)
        random.seed(opts.random_seed)
        
        # Setup model
        self.model = DeeplabV3Plus(opts)
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        
        # Setup metrics
        self.metrics = StreamSegMetrics(opts.num_classes)
        
        # Setup optimizer
        self.optimizer = self.get_optimizer(self.model, opts.lr, opts.lr * 0.1)
        
        # Setup scheduler
        self.scheduler = CyclicLR(self.optimizer, base_lr=opts.lr * 0.1, max_lr=opts.lr,
                                  step_size_up=1000, mode='triangular2')
        
        # Setup criterion
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        
        # Setup wandb
        wandb.init(project="segmentation_project", config=vars(opts))
        wandb.watch(self.model, log="all")
        
        self.denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_optimizer(self, model, lr_classifier, lr_backbone):
        params = [
            {'params': model.module.backbone.parameters(), 'lr': lr_backbone},
            {'params': model.module.classifier.parameters(), 'lr': lr_classifier}
        ]
        return SGD(params, momentum=0.9, weight_decay=self.opts.weight_decay)

    def unfreeze_layers(self, num_layers_to_unfreeze):
        for param in self.model.parameters():
            param.requires_grad = False
        
        layers_to_unfreeze = list(self.model.named_parameters())[-num_layers_to_unfreeze:]
        for name, param in layers_to_unfreeze:
            param.requires_grad = True
            print(f'Unfrozen parameter: {name}')

    def mixup_data(self, x, y, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def knowledge_distillation_loss(self, outputs, labels, teacher_outputs, T=2.0, alpha=0.5):
        kd_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                                 F.softmax(teacher_outputs/T, dim=1)) * (T * T)
        ce_loss = F.cross_entropy(outputs, labels)
        return alpha * kd_loss + (1. - alpha) * ce_loss

    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        pbar = tqdm(train_loader)
        
        for images, labels in pbar:
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)
            
            # Mixup
            images, labels_a, labels_b, lam = self.mixup_data(images, labels)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Knowledge Distillation
            with torch.no_grad():
                teacher_outputs = self.teacher_model(images)
            
            loss = lam * self.knowledge_distillation_loss(outputs, labels_a, teacher_outputs) + \
                   (1 - lam) * self.knowledge_distillation_loss(outputs, labels_b, teacher_outputs)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            pbar.set_description(f"Loss: {loss.item():.4f}")
        
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)
                
                outputs = self.model(images)
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = labels.cpu().numpy()
                
                self.metrics.update(targets, preds)
        
        score = self.metrics.get_results()
        return score

    def save_ckpt(self, path):
        torch.save({
            "model_state": self.model.module.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_score": self.best_score,
        }, path)
        print("Model saved as %s" % path)

    def load_ckpt(self, path):
        if not os.path.isfile(path):
            print(f"Checkpoint not found: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.model.module.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.best_score = checkpoint['best_score']
        print(f"Model restored from {path}")

    def train_and_validate(self, train_loader, val_loader):
        self.best_score = 0.0
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.eval()
        
        for epoch in range(self.opts.epochs):
            # Gradually unfreeze layers
            if epoch == 0:
                self.unfreeze_layers(len(list(self.model.module.classifier.parameters())))
            elif epoch == self.opts.epochs // 2:
                self.unfreeze_layers(50)  # Unfreeze last 50 layers
            
            train_loss = self.train(train_loader)
            val_score = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}/{self.opts.epochs}, Train Loss: {train_loss:.4f}, Val mIoU: {val_score['Mean IoU']:.4f}")
            
            wandb.log({
                "Epoch": epoch + 1,
                "Train Loss": train_loss,
                "Val mIoU": val_score['Mean IoU']
            })
            
            if val_score['Mean IoU'] > self.best_score:
                self.best_score = val_score['Mean IoU']
                self.save_ckpt(f'checkpoints/best_model_epoch{epoch+1}.pth')

if __name__ == "__main__":
    # Setup your argparse here to get opts
    # ...

    trainer = Trainer(opts)
    
    # Setup your dataloaders here
    train_loader = ...
    val_loader = ...
    
    trainer.train_and_validate(train_loader, val_loader)