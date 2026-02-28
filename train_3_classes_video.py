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

from demamba import XCLIP_DeMamba
from dataset_video import TrainDataset, ValDataset, TestDataset, DATA_DIR, TEST_DATA_DIR
from muon import MuonWithAuxAdam

# Set precision and seed
torch.set_float32_matmul_precision('high')
L.seed_everything(42)

# Configuration
BATCH_SIZE = 4
DEVICE_COUNT = 8
EPOCHS = 50
NUM_WORKERS = 4


# WandB setup
import wandb
wandb.login(key="")


def split_muon_params(model):
    muon_params = []
    adam_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            muon_params.append(p)
        else:
            adam_params.append(p)

    return muon_params, adam_params


class ModelLightning(L.LightningModule):
    def __init__(self, model_name='xclip-base-patch16'):
        super().__init__()
        self.model = XCLIP_DeMamba(model_name, class_num=2)
        # self.model = torch.compile(self.model, mode='max-autotune')
        
        # Criterion
        self.train_criterion = torch.nn.CrossEntropyLoss()
        self.val_criterion = torch.nn.CrossEntropyLoss()
        # self.fix_val_criterion = torch.nn.CrossEntropyLoss()
        # self.random_val_criterion = torch.nn.CrossEntropyLoss()
        # self.train_criterion = LabelSmoothingCrossEntropy(smoothing=0.01)
        # self.val_criterion = LabelSmoothingCrossEntropy(smoothing=0.01)
        
        # Train metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=2)

        # Val metrics
        self.val_acc = Accuracy(task='multiclass', num_classes=2)
        self.val_conf_matrix = ConfusionMatrix(task='multiclass', num_classes=2)
        self.val_mcc = MatthewsCorrCoef(task='multiclass', num_classes=2)
        self.val_acc_by_source = {}
        for dataset_name in os.listdir(DATA_DIR):
            if dataset_name in [
                # "eidon-video",
                # "semisynthetic-video",
                # "fakeparts-faceswap",
                # "dfd-real",
                # "dfd-fake",
                "celeb-df-v2",
                "celeb-df-v1",
                # "rtfs-10k-inswapper",
                "rtfs-10k-uniface",
                # "rtfs-10k-original_videos",
                "UADFV-fake",
                "UADFV-real",
                # Thêm các dataset không có trong round này
                "evalcrafter-t2v",
                "text-2-video-human-preferences-moonvalley-marey",
                "lovora-real"
            ]:
                continue
            self.val_acc_by_source[dataset_name.replace('.', '_')] = Accuracy(task='multiclass', num_classes=2)
        self.val_acc_by_source = nn.ModuleDict(self.val_acc_by_source)
        
        # Test metrics
        self.test_criterion = torch.nn.CrossEntropyLoss()
        self.test_acc = Accuracy(task='multiclass', num_classes=2)
        self.test_mcc = MatthewsCorrCoef(task='multiclass', num_classes=2)
        self.test_acc_by_source = {}
        for dataset_name in os.listdir(TEST_DATA_DIR):
            self.test_acc_by_source[dataset_name.replace('.', '_')] = Accuracy(task='multiclass', num_classes=2)
        self.test_acc_by_source = nn.ModuleDict(self.test_acc_by_source)

    def train_dataloader(self):
        train_dataset = TrainDataset(
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
        val_dataset = ValDataset()
        return DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            num_workers=NUM_WORKERS, 
            shuffle=False, 
            drop_last=False,
        )
    
    def test_dataloader(self):
        test_dataset = TestDataset()
        return DataLoader(
            test_dataset, 
            batch_size=BATCH_SIZE, 
            num_workers=NUM_WORKERS, 
            shuffle=False, 
            drop_last=False,
        )
        
    def training_step(self, batch, batch_idx):
        x, y, w = batch
        y = y.long()
        # Convert weights to float32 for loss computation
        w = w.to(dtype=torch.float32)
        pred = self.model(x)
        loss = self.train_criterion(pred, y)
        loss = (loss * w).sum() / w.sum().clamp(min=1.0)  # Weighted loss
        self.log('train_loss', loss, sync_dist=True, on_step=True, on_epoch=False)
        
        # metric: Only update with samples that have non-zero weight to prevent skewing metrics with ignored samples
        mask = w > 0.5
        if mask.any():
            self.train_acc.update(pred[mask], y[mask])
        # self.train_acc.update(pred, y)
        return loss
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, dataset_name, w = batch
        y = y.long()
        # Add w to validation step for weighted metrics
        w = w.float()
        pred = self.model(x)
        loss = self.val_criterion(pred, y)
        loss = (loss * w).sum() / w.sum().clamp(min=1.0)  # Weighted loss
        self.log('val_loss', loss, sync_dist=True, on_step=False, on_epoch=True)
        # self.val_acc.update(pred, y)
        # self.val_conf_matrix.update(pred, y)
        # self.val_mcc.update(pred, y)
        # for idx in range(len(dataset_name)):
        #     source = dataset_name[idx].replace('.', '_')
        #     self.val_acc_by_source[source].update(pred[idx].unsqueeze(0), y[idx].unsqueeze(0))
        
        mask = w > 0.5
        if mask.any():
            self.val_acc.update(pred[mask], y[mask])
            self.val_conf_matrix.update(pred[mask], y[mask])
            self.val_mcc.update(pred[mask], y[mask])
            for idx in range(len(dataset_name)):
                if w[idx] > 0.5:
                    source = dataset_name[idx].replace('.', '_')
                    self.val_acc_by_source[source].update(pred[idx].unsqueeze(0), y[idx].unsqueeze(0))
                
    def test_step(self, batch, batch_idx):
        x, y, dataset_name = batch
        y = y.long()
        pred = self.model(x)
        loss = self.test_criterion(pred, y)
        self.log('test/loss', loss, sync_dist=True, on_step=False, on_epoch=True)
        self.log('test/acc', self.test_acc(pred, y), sync_dist=True, on_step=False, on_epoch=True)
        self.log('test/mcc', self.test_mcc(pred, y), sync_dist=True, on_step=False, on_epoch=True)
        # self.test_acc.update(pred, y)
        # self.test_mcc.update(pred, y)
        for idx in range(len(dataset_name)):
            source = dataset_name[idx].replace('.', '_')
            # self.test_acc_by_source[source].update(pred[idx].unsqueeze(0), y[idx].unsqueeze(0))
            self.log(f'test/acc_by_source/{source}', self.test_acc(pred[idx].unsqueeze(0), y[idx].unsqueeze(0)), sync_dist=True, on_step=False, on_epoch=True)
        
        return loss
    
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
            self.log(f'val/acc_by_source/{source}', acc, sync_dist=True)

        # for source in self.fix_val_acc_by_source.keys():
        #     norm_acc = fix_val_acc_by_source[source]
        #     weight = (1 - max(norm_acc.item(), 0.8)) // 0.05 + 1
        #     # weight = 8 if norm_acc.item() < 0.95 else 4
        #     self.train_weights[source] = weight
        #     # self.print(f"Train weight for {source}: {weight}")

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
        # optimizer = torch.optim.AdamW(
        #     self.model.parameters(), 
        #     lr=1e-5, 
        #     weight_decay=1e-2,
        # )
        
        muon_params, adam_params = split_muon_params(self.model)
        param_groups = [
            dict(
                params=muon_params,
                use_muon=True,
                lr=3e-5,
                momentum=0.95,
                weight_decay=1e-2,
            ),
            dict(
                params=adam_params,
                use_muon=False,
                lr=1e-5,
                betas=(0.9, 0.95),
                weight_decay=1e-2,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=100,
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10000,
            eta_min=1e-6,
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
    model_name = "microsoft/xclip-base-patch16-16-frames"
    exp_name = "3_classes_video_demamba_muon_r12_loss_ce_wo_lovora_real_2802"
    # ckpt_path = "/mnt/bitmind/34/bitmind_video/microsoft_xclip-base-patch16-16-frames_3_classes_video_5500_muon_851ad848/checkpoints/last.ckpt"
    ckpt_path = None
    if ckpt_path and os.path.exists(ckpt_path):
        _id = ckpt_path.split("/")[-3].split("_")[-1]
    else:
        _id = str(uuid.uuid4())[:8]
    model = ModelLightning(
        model_name=model_name,
    )
    exp_name = model_name.replace('/', '_') + "_" + exp_name
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        save_last=True, 
        every_n_epochs=1, 
        filename="{epoch:02d}", 
        save_top_k=-1
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Logger
    wandb_logger = WandbLogger(project="bitmind_video", name=exp_name, id=exp_name + "_" + _id)
    
    # Trainer
    trainer = L.Trainer(
        precision='bf16-mixed',
        max_epochs=EPOCHS,
        devices=8,
        accelerator="gpu",
        strategy=DDPStrategy(static_graph=True),
        inference_mode=True,
        callbacks=[checkpoint_callback, lr_monitor], 
        logger=wandb_logger,
        num_sanity_val_steps=10,
        default_root_dir='checkpoints',
        reload_dataloaders_every_n_epochs=1,
        # fast_dev_run=True
    )
    
    # Training
    if ckpt_path and os.path.exists(ckpt_path):
        trainer.fit(
            model,
            ckpt_path=ckpt_path
        )
    else:
        trainer.fit(
            model,
        )


if __name__ == '__main__':    
    main()