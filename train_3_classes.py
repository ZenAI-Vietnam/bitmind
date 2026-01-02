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

from model import Model
from dataset import TrainDataset, ValDataset

# Set precision and seed
torch.set_float32_matmul_precision('high')
L.seed_everything(42)

# Configuration
BATCH_SIZE = 64
DEVICE_COUNT = 8
EPOCHS = 100
NUM_WORKERS = 96


# WandB setup
import wandb
wandb.login(key="")


class ModelLightning(L.LightningModule):
    def __init__(self, model_name='convnextv2_tiny.fcmae_ft_in22k_in1k_384'):
        super().__init__()
        self.model = Model(model_name, num_classes=2)
        # self.model = torch.compile(self.model, mode='max-autotune')
        
        # Criterion
        # self.train_criterion = torch.nn.CrossEntropyLoss()
        # self.val_criterion = torch.nn.CrossEntropyLoss()
        self.train_criterion = LabelSmoothingCrossEntropy(smoothing=0.01)
        self.val_criterion = LabelSmoothingCrossEntropy(smoothing=0.01)
        
        # Train metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=2)

        # Val metrics
        self.val_acc = Accuracy(task='multiclass', num_classes=2)
        self.val_conf_matrix = ConfusionMatrix(task='multiclass', num_classes=2)
        self.val_mcc = MatthewsCorrCoef(task='multiclass', num_classes=2)
        self.val_acc_by_source = {}
        
        self.train_weights = {}
        
        # Acc by source
        ROOT_DIR = "/mnt/video/video/datasets"
        for dataset_name in os.listdir(ROOT_DIR):
            self.val_acc_by_source[dataset_name.replace('.', '_')] = Accuracy(task='multiclass', num_classes=2)
        self.val_acc_by_source = nn.ModuleDict(self.val_acc_by_source)

    def train_dataloader(self):
        train_dataset = TrainDataset()
        return DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            num_workers=NUM_WORKERS, 
            shuffle=True, 
            pin_memory=True, 
            drop_last=True,
        )

    def val_dataloader(self):
        val_dataset = ValDataset()
        return DataLoader(
                val_dataset, 
                batch_size=BATCH_SIZE, 
                num_workers=NUM_WORKERS, 
                shuffle=False, 
                drop_last=False,
            )
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        pred = self.model(x)
        loss = self.train_criterion(pred, y)
        self.log('train_loss', loss, sync_dist=True, on_step=True, on_epoch=False)
        self.train_acc.update(pred, y)
        return loss
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, dataset_name = batch
        y = y.long()
        pred = self.model(x)

        loss = self.val_criterion(pred, y)
        self.log('val_loss', loss, sync_dist=True, on_step=False, on_epoch=True)
        self.val_acc.update(pred, y)
        self.val_conf_matrix.update(pred, y)
        self.val_mcc.update(pred, y)
        for idx in range(len(dataset_name)):
            self.val_acc_by_source[dataset_name[idx].replace('.', '_')].update(pred[idx].unsqueeze(0), y[idx].unsqueeze(0))
    
    def on_validation_epoch_end(self):
        self.logger.log_image(
            key='confusion_matrix',
            images=[self.val_conf_matrix.plot()[0]],
            caption=[f'Validation Confusion Matrix - Epoch {self.current_epoch}'],
        )

        # Compute and log metrics
        val_acc = self.val_acc.compute()
        val_mcc = self.val_mcc.compute()
        val_acc_by_source = {}
        for source, acc in self.val_acc_by_source.items():
            val_acc_by_source[source] = acc.compute()

        self.print(f"[Epoch {self.current_epoch}] Val Acc: {val_acc:.4f} | "
                   f"Val MCC: {val_mcc:.4f}")

        self.log('val/acc', val_acc, sync_dist=True)
        self.log('val/mcc', val_mcc, sync_dist=True)

        for source, acc in val_acc_by_source.items():
            self.print(f"Val Acc for {source}: {acc:.4f}")
            self.log(f'val/acc_by_source_{source}', acc, sync_dist=True)

        # Reset metrics
        self.val_acc.reset()
        self.val_conf_matrix.reset()
        self.val_mcc.reset()
        for source in self.val_acc_by_source.keys():
            self.val_acc_by_source[source].reset()

    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc.compute(), sync_dist=True)
        self.train_acc.reset()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=1e-4, 
            weight_decay=1e-2,
        )

        # Linear warmup
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=100,
        )

        multi_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[40, 70],
            gamma=0.5,
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, multi_scheduler],
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
    model_name = "convnextv2_base.fcmae_ft_in22k_in1k"
    exp_name = "overfit_10000_ls0.01"
    model = ModelLightning.load_from_checkpoint(
        "/mnt/video/bitmind_video_3_classes/convnextv2_base.fcmae_ft_in22k_in1k_overfit_10000_ls10_90163d64/checkpoints/epoch=99.ckpt",
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
    wandb_logger = WandbLogger(project="bitmind_video_3_classes", name=exp_name, id=exp_name + "_" + str(uuid.uuid4())[:8])
    
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
        num_sanity_val_steps=-1,
        default_root_dir='checkpoints',
        # reload_dataloaders_every_n_epochs=1,
        # fast_dev_run=True
    )
    
    # Training
    trainer.fit(
        model    
    )


if __name__ == '__main__':    
    main()