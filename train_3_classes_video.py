import os
import uuid
import time
import json
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import Accuracy, ConfusionMatrix, MatthewsCorrCoef
from lightning.pytorch.strategies import DDPStrategy
from timm.loss import LabelSmoothingCrossEntropy

import warnings
warnings.filterwarnings("ignore")

from model_video import XCLIP_DeMamba
from dataset_video import TrainDataset, FixValDataset, RandomValDataset

# Set precision and seed
torch.set_float32_matmul_precision('high')
L.seed_everything(42)

# Configuration
BATCH_SIZE = 8
DEVICE_COUNT = 8
EPOCHS = 50
NUM_WORKERS = 4


# WandB setup
import wandb
wandb.login(key="")


class ModelLightning(L.LightningModule):
    def __init__(self, model_name='xclip-base-patch16'):
        super().__init__()
        self.model = XCLIP_DeMamba(model_name, class_num=2)
        # self.model = torch.compile(self.model, mode='max-autotune')
        
        # Criterion
        # self.train_criterion = torch.nn.CrossEntropyLoss()
        # self.fix_val_criterion = torch.nn.CrossEntropyLoss()
        # self.random_val_criterion = torch.nn.CrossEntropyLoss()
        self.train_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        self.fix_val_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        self.random_val_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        # Train metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=2)

        # Fix val metrics
        self.fix_val_acc = Accuracy(task='multiclass', num_classes=2)
        self.fix_val_conf_matrix = ConfusionMatrix(task='multiclass', num_classes=2)
        self.fix_val_mcc = MatthewsCorrCoef(task='multiclass', num_classes=2)
        self.fix_val_acc_by_source = {}
        self.fix_val_acc_by_level = {}
        for i in range(1):
            self.fix_val_acc_by_level[str(i)] = Accuracy(task='multiclass', num_classes=2)
        self.fix_val_acc_by_level = nn.ModuleDict(self.fix_val_acc_by_level)

        # Random val metrics
        self.random_val_acc = Accuracy(task='multiclass', num_classes=2)
        self.random_val_conf_matrix = ConfusionMatrix(task='multiclass', num_classes=2)
        self.random_val_mcc = MatthewsCorrCoef(task='multiclass', num_classes=2)
        self.random_val_acc_by_source = {}
        self.random_val_acc_by_level = {}
        for i in range(1):
            self.random_val_acc_by_level[str(i)] = Accuracy(task='multiclass', num_classes=2)
        self.random_val_acc_by_level = nn.ModuleDict(self.random_val_acc_by_level)
        
        self.train_weights = {}
        
        self.train_dataset_file = 'all_video.jsonl'
        self.val_dataset_file = 'val_video.jsonl'
        
        # Acc by source
        sources = set()
        for line in open(self.val_dataset_file, 'r'):
            data = json.loads(line)
            sources.add(data['dataset'])
        for source in sources:
            source = source.replace('.', '_')
            self.fix_val_acc_by_source[source] = Accuracy(task='multiclass', num_classes=2)
            self.random_val_acc_by_source[source] = Accuracy(task='multiclass', num_classes=2)
        self.fix_val_acc_by_source = nn.ModuleDict(self.fix_val_acc_by_source)
        self.random_val_acc_by_source = nn.ModuleDict(self.random_val_acc_by_source)

    def train_dataloader(self):
        train_dataset = TrainDataset(
            dataset_file=self.train_dataset_file, 
            # base_transforms=self.model.get_transforms(is_training=False),
            label_map={0: 0, 1: 1, 2: 1},
            weights=self.train_weights
        )
        return DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            num_workers=NUM_WORKERS, 
            shuffle=True, 
            pin_memory=True, 
            drop_last=True,
        )

    def val_dataloader(self):
        fix_val_dataset = FixValDataset(
            dataset_file=self.val_dataset_file,
            # base_transforms=self.model.get_transforms(is_training=False),
            label_map={0: 0, 1: 1, 2: 1},
        )
        random_val_dataset = RandomValDataset(
            dataset_file=self.train_dataset_file, 
            # base_transforms=self.model.get_transforms(is_training=False),
            label_map={0: 0, 1: 1, 2: 1},
        )
        return [
            DataLoader(
                fix_val_dataset, 
                batch_size=BATCH_SIZE, 
                num_workers=NUM_WORKERS, 
                shuffle=False, 
                drop_last=False,
            ),
            DataLoader(
                random_val_dataset, 
                batch_size=BATCH_SIZE, 
                num_workers=NUM_WORKERS,    
                shuffle=False, 
                drop_last=False,
            ),
        ]
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        pred = self.model(x)
        loss = self.train_criterion(pred, y)
        self.log('train_loss', loss, sync_dist=True, on_step=True, on_epoch=False)
        self.train_acc.update(pred, y)
        return loss
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, sources, levels = batch
        y = y.long()
        pred = self.model(x)
        if dataloader_idx == 0:
            loss = self.fix_val_criterion(pred, y)
            self.log('fix_val_loss', loss, sync_dist=True, on_step=False, on_epoch=True)
            self.fix_val_acc.update(pred, y)
            self.fix_val_conf_matrix.update(pred, y)
            self.fix_val_mcc.update(pred, y)
            for idx in range(len(sources)):
                level_idx = levels[idx].item()
                self.fix_val_acc_by_level[str(level_idx)]
                source = sources[idx].replace('.', '_')
                if source not in self.fix_val_acc_by_source.keys():
                    self.fix_val_acc_by_source[source].update(
                        {
                            source: Accuracy(task='multiclass', num_classes=2)
                        }
                    )

                self.fix_val_acc_by_level[str(level_idx)].update(pred[idx].unsqueeze(0), y[idx].unsqueeze(0))
                self.fix_val_acc_by_source[source].update(pred[idx].unsqueeze(0), y[idx].unsqueeze(0))
        else:
            loss = self.random_val_criterion(pred, y)
            self.log('random_val_loss', loss, sync_dist=True, on_step=False, on_epoch=True)
            self.random_val_acc.update(pred, y)
            self.random_val_conf_matrix.update(pred, y)
            self.random_val_mcc.update(pred, y)
            for idx in range(len(sources)):
                source = sources[idx].replace('.', '_')
                level_idx = levels[idx].item()
                self.random_val_acc_by_level[str(level_idx)]
                if source not in self.random_val_acc_by_source.keys():
                    self.random_val_acc_by_source[source].update(
                        {
                            source: Accuracy(task='multiclass', num_classes=2)
                        }
                    )
                self.random_val_acc_by_level[str(level_idx)].update(pred[idx].unsqueeze(0), y[idx].unsqueeze(0))
                self.random_val_acc_by_source[source].update(pred[idx].unsqueeze(0), y[idx].unsqueeze(0))
    
    def on_validation_epoch_end(self):
        self.logger.log_image(
            key='confusion_matrix',
            images=[self.fix_val_conf_matrix.plot()[0], self.random_val_conf_matrix.plot()[0]],
            caption=[f'Fix Validation Confusion Matrix - Epoch {self.current_epoch}', f'Random Validation Confusion Matrix - Epoch {self.current_epoch}'],
        )

        # Compute and log metrics
        fix_val_acc = self.fix_val_acc.compute()
        random_val_acc = self.random_val_acc.compute()
        fix_val_mcc = self.fix_val_mcc.compute()
        random_val_mcc = self.random_val_mcc.compute()
        fix_val_acc_by_source = {}
        for source, acc in self.fix_val_acc_by_source.items():
            fix_val_acc_by_source[source] = acc.compute()
        fix_val_acc_by_level = {}
        for level, acc in self.fix_val_acc_by_level.items():
            fix_val_acc_by_level[str(level)] = acc.compute()
        random_val_acc_by_source = {}
        for source, acc in self.random_val_acc_by_source.items():
            random_val_acc_by_source[source] = acc.compute()
        random_val_acc_by_level = {}
        for level, acc in self.random_val_acc_by_level.items():
            random_val_acc_by_level[str(level)] = acc.compute()

        self.print(f"[Epoch {self.current_epoch}] Fix Val Acc: {fix_val_acc:.4f} | "
                   f"Random Val Acc: {random_val_acc:.4f} | "
                   f"Fix Val MCC: {fix_val_mcc:.4f} | "
                   f"Random Val MCC: {random_val_mcc:.4f}")

        self.log('fix_val/acc', fix_val_acc, sync_dist=True)
        self.log('random_val/acc', random_val_acc, sync_dist=True)
        self.log('fix_val/mcc', fix_val_mcc, sync_dist=True)
        self.log('random_val/mcc', random_val_mcc, sync_dist=True)

        for source, acc in fix_val_acc_by_source.items():
            self.print(f"Fix Val Acc for {source}: {acc:.4f}")
            self.log(f'fix_val/acc_by_source_{source}', acc, sync_dist=True)
        for level, acc in fix_val_acc_by_level.items():
            self.print(f"Fix Val Acc for level {level}: {acc:.4f}")
            self.log(f'fix_val/acc_by_level_{str(level)}', acc, sync_dist=True)
        for source, acc in random_val_acc_by_source.items():
            self.print(f"Random Val Acc for {source}: {acc:.4f}")
            self.log(f'random_val/acc_by_source_{source}', acc, sync_dist=True)
        for level, acc in random_val_acc_by_level.items():
            self.print(f"Random Val Acc for level {level}: {acc:.4f}")
            self.log(f'random_val/acc_by_level_{str(level)}', acc, sync_dist=True)

        # for source in self.fix_val_acc_by_source.keys():
        #     norm_acc = fix_val_acc_by_source[source]
        #     weight = (1 - max(norm_acc.item(), 0.8)) // 0.05 + 1
        #     # weight = 8 if norm_acc.item() < 0.95 else 4
        #     self.train_weights[source] = weight
        #     # self.print(f"Train weight for {source}: {weight}")

        # Reset metrics
        self.fix_val_acc.reset()
        self.fix_val_conf_matrix.reset()
        self.fix_val_mcc.reset()
        for source in self.fix_val_acc_by_source.keys():
            self.fix_val_acc_by_source[source].reset()
        for level in self.fix_val_acc_by_level.keys():
            self.fix_val_acc_by_level[str(level)].reset()
        self.random_val_acc.reset()
        self.random_val_conf_matrix.reset()
        self.random_val_mcc.reset()
        for source in self.random_val_acc_by_source.keys():
            self.random_val_acc_by_source[source].reset()
        for level in self.random_val_acc_by_level.keys():
            self.random_val_acc_by_level[level].reset()

    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc.compute(), sync_dist=True)
        self.train_acc.reset()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=1e-5, 
            weight_decay=1e-2,
        )

        # Linear warmup
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=100,
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10000,
            eta_min=5e-6,
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[100],
        )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def main():
    model_name = "xclip-base-patch16"
    exp_name = "3_classes_overfit_10000_extended_data"
    model = ModelLightning(
        model_name=model_name,
    )
    exp_name = model_name + "_" + exp_name
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        save_last=True, 
        every_n_epochs=1, 
        filename="{epoch:02d}", 
        save_top_k=-1
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Logger
    wandb_logger = WandbLogger(project="bitmind_video_3_classes_video", name=exp_name, id=exp_name + "_" + str(uuid.uuid4())[:8])
    
    # Trainer
    trainer = L.Trainer(
        precision='bf16-mixed',
        max_epochs=EPOCHS,
        devices=DEVICE_COUNT,
        accelerator="gpu",
        strategy=DDPStrategy(static_graph=True),
        inference_mode=True,
        callbacks=[checkpoint_callback, lr_monitor], 
        logger=wandb_logger,
        num_sanity_val_steps=0,
        default_root_dir='checkpoints',
        reload_dataloaders_every_n_epochs=1,
        # fast_dev_run=True
    )
    
    # Training
    trainer.fit(
        model,
    )


if __name__ == '__main__':    
    main()