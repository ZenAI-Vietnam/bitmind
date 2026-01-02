import os
import uuid
import time
import json
import logging
import sys
from typing import Callable
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from torchmetrics import Accuracy, ConfusionMatrix, MatthewsCorrCoef
from timm.loss import LabelSmoothingCrossEntropy
from torch.cuda.amp import autocast, GradScaler

import warnings
warnings.filterwarnings("ignore")

from model_video import XCLIP_DeMamba
from dataset_video import TrainDataset, FixValDataset, RandomValDataset

# Set precision and seed
torch.set_float32_matmul_precision('high')
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Configuration
BATCH_SIZE = 8
DEVICE_COUNT = 8
EPOCHS = 50
NUM_WORKERS = 4


def setup_ddp():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        init_process_group(backend='nccl', rank=rank, world_size=world_size)
        
        return rank, world_size, local_rank
    else:
        # Single GPU or CPU
        return 0, 1, 0


def cleanup_ddp():
    """Cleanup distributed training"""
    if dist.is_initialized():
        destroy_process_group()


def create_model(model_name='convnextv2_tiny.fcmae_ft_in22k_in1k_384', device=None):
    """Create and return the model"""
    model = XCLIP_DeMamba(model_name, class_num=2)
    if device is not None:
        model = model.to(device)
    return model


def create_datasets(model, train_weights=None):
    """Create train and validation datasets"""
    train_dataset_file = 'all_video.jsonl'
    val_dataset_file = 'val_video.jsonl'
    
    train_dataset = TrainDataset(
        dataset_file=train_dataset_file, 
        label_map={0: 0, 1: 1, 2: 1},
        weights=train_weights if train_weights else {}
    )
    
    fix_val_dataset = FixValDataset(
        dataset_file=val_dataset_file,
        label_map={0: 0, 1: 1, 2: 1},
    )
    
    random_val_dataset = RandomValDataset(
        dataset_file=train_dataset_file, 
        label_map={0: 0, 1: 1, 2: 1},
    )
    
    return train_dataset, fix_val_dataset, random_val_dataset


def create_dataloaders(train_dataset, fix_val_dataset, random_val_dataset, 
                       rank=0, world_size=1, is_distributed=False):
    """Create data loaders with optional distributed sampling"""
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=True
        )
        fix_val_sampler = DistributedSampler(
            fix_val_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=False
        )
        random_val_sampler = DistributedSampler(
            random_val_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=False
        )
    else:
        train_sampler = None
        fix_val_sampler = None
        random_val_sampler = None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        shuffle=(train_sampler is None), 
        pin_memory=True, 
        drop_last=True,
        sampler=train_sampler
    )
    
    fix_val_loader = DataLoader(
        fix_val_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        shuffle=False, 
        drop_last=False,
        sampler=fix_val_sampler
    )
    
    random_val_loader = DataLoader(
        random_val_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,    
        shuffle=False, 
        drop_last=False,
        sampler=random_val_sampler
    )
    
    return train_loader, fix_val_loader, random_val_loader


def create_metrics(val_dataset_file):
    """Create metrics dictionaries"""
    # Get sources from validation dataset
    sources = set()
    for line in open(val_dataset_file, 'r'):
        data = json.loads(line)
        sources.add(data['dataset'])
    
    # Fix val metrics
    fix_val_acc = Accuracy(task='multiclass', num_classes=2)
    fix_val_conf_matrix = ConfusionMatrix(task='multiclass', num_classes=2)
    fix_val_mcc = MatthewsCorrCoef(task='multiclass', num_classes=2)
    fix_val_acc_by_source = {}
    fix_val_acc_by_level = {}
    for i in range(4):
        fix_val_acc_by_level[str(i)] = Accuracy(task='multiclass', num_classes=2)
    for source in sources:
        source_clean = source.replace('.', '_')
        fix_val_acc_by_source[source_clean] = Accuracy(task='multiclass', num_classes=2)
    
    # Random val metrics
    random_val_acc = Accuracy(task='multiclass', num_classes=2)
    random_val_conf_matrix = ConfusionMatrix(task='multiclass', num_classes=2)
    random_val_mcc = MatthewsCorrCoef(task='multiclass', num_classes=2)
    random_val_acc_by_source = {}
    random_val_acc_by_level = {}
    for i in range(4):
        random_val_acc_by_level[str(i)] = Accuracy(task='multiclass', num_classes=2)
    for source in sources:
        source_clean = source.replace('.', '_')
        random_val_acc_by_source[source_clean] = Accuracy(task='multiclass', num_classes=2)
    
    return {
        'fix_val_acc': fix_val_acc,
        'fix_val_conf_matrix': fix_val_conf_matrix,
        'fix_val_mcc': fix_val_mcc,
        'fix_val_acc_by_source': fix_val_acc_by_source,
        'fix_val_acc_by_level': fix_val_acc_by_level,
        'random_val_acc': random_val_acc,
        'random_val_conf_matrix': random_val_conf_matrix,
        'random_val_mcc': random_val_mcc,
        'random_val_acc_by_source': random_val_acc_by_source,
        'random_val_acc_by_level': random_val_acc_by_level,
    }


def reset_metrics(metrics):
    """Reset all metrics"""
    metrics['fix_val_acc'].reset()
    metrics['fix_val_conf_matrix'].reset()
    metrics['fix_val_mcc'].reset()
    for acc in metrics['fix_val_acc_by_source'].values():
        acc.reset()
    for acc in metrics['fix_val_acc_by_level'].values():
        acc.reset()
    metrics['random_val_acc'].reset()
    metrics['random_val_conf_matrix'].reset()
    metrics['random_val_mcc'].reset()
    for acc in metrics['random_val_acc_by_source'].values():
        acc.reset()
    for acc in metrics['random_val_acc_by_level'].values():
        acc.reset()


def train_epoch(model, train_loader, optimizer, scheduler, criterion, metrics, 
                device, scaler, global_step, rank=0, world_size=1):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=False)
    for batch_idx, (x, y) in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            pred = model(x)
            loss = criterion(pred, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Update metrics
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update tqdm with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Sync metrics across processes
    if world_size > 1:
        # Average loss across processes
        loss_tensor = torch.tensor(avg_loss).to(device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size
    
    if rank == 0:
        logger = logging.getLogger(__name__)
        logger.info(f"Train Loss: {avg_loss:.4f}")
    
    return avg_loss, global_step


def validate(model, val_loader, metrics_dict, device, 
             dataloader_name='fix_val', rank=0, world_size=8):
    """Validate on a single dataloader"""
    model.eval()
    
    with torch.no_grad():
        for x, y, sources, levels in tqdm(val_loader, total=len(val_loader), desc=f"Validating {dataloader_name}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            
            with autocast():
                pred = model(x)
            
            # Update metrics
            metrics_dict[f'{dataloader_name}_acc'].update(pred, y)
            metrics_dict[f'{dataloader_name}_conf_matrix'].update(pred, y)
            metrics_dict[f'{dataloader_name}_mcc'].update(pred, y)
            
            # Update per-source and per-level metrics
            for idx in range(len(sources)):
                level_idx = levels[idx].item()
                source = sources[idx].replace('.', '_')
                
                if source in metrics_dict[f'{dataloader_name}_acc_by_source']:
                    metrics_dict[f'{dataloader_name}_acc_by_source'][source].update(
                        pred[idx].unsqueeze(0), y[idx].unsqueeze(0)
                    )
                if str(level_idx) in metrics_dict[f'{dataloader_name}_acc_by_level']:
                    # print(f"Rank {rank} {dataloader_name} level {level_idx} update")
                    metrics_dict[f'{dataloader_name}_acc_by_level'][str(level_idx)].update(
                        pred[idx].unsqueeze(0), y[idx].unsqueeze(0)
                    )


def validate_epoch(model, fix_val_loader, random_val_loader, 
                   metrics, device, epoch, global_step, rank=0):
    """Run validation"""
    # Fix validation
    validate(
        model, fix_val_loader, 
        {'fix_val_acc': metrics['fix_val_acc'],
         'fix_val_conf_matrix': metrics['fix_val_conf_matrix'],
         'fix_val_mcc': metrics['fix_val_mcc'],
         'fix_val_acc_by_source': metrics['fix_val_acc_by_source'],
         'fix_val_acc_by_level': metrics['fix_val_acc_by_level']},
        device, 'fix_val',
        rank
    )
    
    # Random validation
    validate(
        model, random_val_loader, 
        {'random_val_acc': metrics['random_val_acc'],
         'random_val_conf_matrix': metrics['random_val_conf_matrix'],
         'random_val_mcc': metrics['random_val_mcc'],
         'random_val_acc_by_source': metrics['random_val_acc_by_source'],
         'random_val_acc_by_level': metrics['random_val_acc_by_level']},
        device, 'random_val',
        rank
    )
    
    # Compute metrics
    fix_val_acc = metrics['fix_val_acc'].compute()
    random_val_acc = metrics['random_val_acc'].compute()
    fix_val_mcc = metrics['fix_val_mcc'].compute()
    random_val_mcc = metrics['random_val_mcc'].compute()
    
    fix_val_acc_by_source = {}
    for source, acc in metrics['fix_val_acc_by_source'].items():
        fix_val_acc_by_source[source] = acc.compute()
    
    fix_val_acc_by_level = {}
    for level, acc in metrics['fix_val_acc_by_level'].items():
        fix_val_acc_by_level[level] = acc.compute()
    
    random_val_acc_by_source = {}
    for source, acc in metrics['random_val_acc_by_source'].items():
        random_val_acc_by_source[source] = acc.compute()
    
    random_val_acc_by_level = {}
    for level, acc in metrics['random_val_acc_by_level'].items():
        random_val_acc_by_level[level] = acc.compute()
    
    # Log metrics
    if rank == 0:
        logger = logging.getLogger(__name__)
        logger.info(f"Fix Val Acc: {fix_val_acc:.4f} | "
              f"Random Val Acc: {random_val_acc:.4f} | "
              f"Fix Val MCC: {fix_val_mcc:.4f} | "
              f"Random Val MCC: {random_val_mcc:.4f}")
        
        log_dict = {
            'fix_val/acc': fix_val_acc.item(),
            'random_val/acc': random_val_acc.item(),
            'fix_val/mcc': fix_val_mcc.item(),
            'random_val/mcc': random_val_mcc.item(),
        }
        
        for source, acc in fix_val_acc_by_source.items():
            logger.info(f"Fix Val Acc for {source}: {acc:.4f}")
            log_dict[f'fix_val/acc_by_source_{source}'] = acc.item()
        
        for level, acc in fix_val_acc_by_level.items():
            logger.info(f"Fix Val Acc for level {level}: {acc:.4f}")
            log_dict[f'fix_val/acc_by_level_{level}'] = acc.item()
        
        for source, acc in random_val_acc_by_source.items():
            logger.info(f"Random Val Acc for {source}: {acc:.4f}")
            log_dict[f'random_val/acc_by_source_{source}'] = acc.item()
        
        for level, acc in random_val_acc_by_level.items():
            logger.info(f"Random Val Acc for level {level}: {acc:.4f}")
            log_dict[f'random_val/acc_by_level_{level}'] = acc.item()


    weights = {}

    if rank == 0:
        for source in fix_val_acc_by_source.keys():
            norm_acc = (fix_val_acc_by_source[source] + random_val_acc_by_source[source]) / 2
            weight = int((1 - max(norm_acc.item(), 0.8)) // 0.05 + 1)
            weights[source] = weight

    if dist.is_initialized():
        obj_list = [weights] if rank == 0 else [None]
        dist.broadcast_object_list(obj_list, src=0)
        weights = obj_list[0]
        dist.barrier()

    reset_metrics(metrics)
    return weights
    

def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir, rank=0):
    """Save model checkpoint"""
    if rank == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'{epoch:02d}.pt')
        # If using DDP, save the underlying model
        model_to_save = model.module if isinstance(model, DDP) else model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, checkpoint_path)
        logger = logging.getLogger(__name__)
        logger.info(f"Checkpoint saved: {checkpoint_path}")


def main():
    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    # print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    is_distributed = world_size > 1
    
    # Device setup
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Model setup
    model_name = "xclip-base-patch16"
    exp_name = "1225"
    exp_name = model_name + "_" + exp_name
    
    model = create_model(model_name, device)
    
    # Wrap model with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # Loss functions
    train_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-6, 
        weight_decay=1e-8,
    )
    
    # Learning rate schedulers
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

    # def load_checkpoint(checkpoint_path):
    #     if checkpoint_path and os.path.exists(checkpoint_path):
    #         checkpoint = torch.load(checkpoint_path)
    #         model.module.load_state_dict(checkpoint['model_state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #         return checkpoint['epoch']
    #     return 0

    # start_epoch = load_checkpoint("")
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Datasets and dataloaders
    train_weights = {}
    train_dataset, fix_val_dataset, random_val_dataset = create_datasets(
        model.module if is_distributed else model, 
        train_weights
    )
    train_loader, fix_val_loader, random_val_loader = create_dataloaders(
        train_dataset, fix_val_dataset, random_val_dataset,
        rank, world_size, is_distributed
    )
    
    # Metrics
    val_dataset_file = 'val_video.jsonl'
    metrics = create_metrics(val_dataset_file)
    for k, v in metrics.items():
        if isinstance(v, dict):
            for m in v.values():
                m.to(device)
        else:
            v.to(device)
    
    # Move metrics to device if needed (torchmetrics works on CPU by default)

    _id = exp_name + "_" + str(uuid.uuid4())[:8]
    outdir = f"checkpoints_pt/bitmind_video_model/{_id}"
    os.makedirs(outdir, exist_ok=True)
    # Setup logging to file
    if rank == 0:
        log_file = f"{outdir}/log.txt"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Training with id: {_id}")
        logger.info(f"Log file: {log_file}")
    else:
        logger = None
    
    # Checkpoint directory
    checkpoint_dir = outdir
    
    # Global step counter for logging
    global_step = 0
    
    # Training loop
    try:
        for epoch in range(EPOCHS):
            if rank == 0:
                logger = logging.getLogger(__name__)
                logger.info(f"=" * 100)
                logger.info(f"Epoch {epoch}: ")
            if is_distributed:
                train_loader.sampler.set_epoch(epoch)
                # fix_val_loader.sampler.set_epoch(epoch)
                # random_val_loader.sampler.set_epoch(epoch)
            
            # Train
            train_loss, global_step = train_epoch(
                model, train_loader, optimizer, scheduler, train_criterion,
                metrics, device, scaler, global_step, rank, world_size
            )
            
            # Validate
            weights = validate_epoch(
                model, fix_val_loader, random_val_loader,
                metrics,
                device, epoch, global_step, rank
            )
            
            # Save checkpoint
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir, rank)
            
            # Reload dataloaders every epoch (recreate datasets with updated weights)
            if epoch < EPOCHS:  # Don't reload on last epoch
                train_dataset, fix_val_dataset, random_val_dataset = create_datasets(
                    model.module if is_distributed else model,
                    weights
                )
                train_loader, fix_val_loader, random_val_loader = create_dataloaders(
                    train_dataset, fix_val_dataset, random_val_dataset,
                    rank, world_size, is_distributed
                )
    
    finally:
        cleanup_ddp()


if __name__ == '__main__':    
    main()

