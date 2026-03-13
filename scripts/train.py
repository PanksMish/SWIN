"""
train.py
========
Main training script for the Swin-FANE framework.

Two-stage training procedure (paper Section 5.1):
  Stage 1: Train the Swin-FANE spatial encoder (frame-level cross-entropy)
  Stage 2: Freeze encoder, train BiLSTM temporal module

Usage:
    python scripts/train.py --config configs/config.yaml

    # Resume training
    python scripts/train.py --config configs/config.yaml \\
                            --resume checkpoints/checkpoint_epoch020.pth

    # Use FER2013 dataset instead of FANE
    python scripts/train.py --config configs/config.yaml --dataset fer2013

Paper reference: Section 5.1 (experimental setup), Section 4.4 (optimisation)
"""

import os
import sys
import argparse
import random
import time
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import yaml

from models.full_framework import build_framework
from data.fane_dataset import build_fane_dataloaders
from data.fer2013_dataset import build_fer2013_dataloaders
from utils.metrics import EmotionMetrics
from utils.logger import TrainingLogger


# ---------------------------------------------------------------------------
# Reproducibility: Fix all random seeds (paper Section 5.1)
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Sets all random seeds for reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    metrics: EmotionMetrics,
    epoch: int,
    log_interval: int = 10,
    logger: TrainingLogger = None,
) -> dict:
    """
    Runs one training epoch over the dataset.

    Args:
        model:        The framework model.
        loader:       Training DataLoader.
        criterion:    Loss function (cross-entropy with label smoothing).
        optimizer:    AdamW optimizer.
        device:       Computation device (CPU/GPU).
        metrics:      EmotionMetrics accumulator.
        epoch:        Current epoch number (for logging).
        log_interval: Print progress every N batches.
        logger:       Optional TrainingLogger instance.

    Returns:
        Training metrics dict.
    """
    model.train()
    metrics.reset()
    total_loss = 0.0
    n_batches  = 0

    for batch_idx, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        masks  = batch.get('mask')
        if masks is not None:
            masks = masks.to(device, non_blocking=True)

        # ---- Forward pass (frame mode: images shape B,C,H,W) ----
        optimizer.zero_grad()
        output = model.forward_frame(images, mask=masks)

        # ---- Cross-entropy loss (paper Equation 14) ----
        loss = criterion(output['logits'], labels)

        # ---- Backward + gradient clip ----
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ---- Accumulate metrics ----
        with torch.no_grad():
            metrics.update(output['probs'], labels, loss=loss.item())
            total_loss += loss.item()
            n_batches  += 1

        # ---- Progress logging ----
        if (batch_idx + 1) % log_interval == 0 and logger:
            logger.info(
                f"  Epoch {epoch} [{batch_idx + 1}/{len(loader)}] "
                f"Loss: {total_loss / n_batches:.4f}"
            )

    return metrics.compute()


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    metrics: EmotionMetrics,
) -> dict:
    """
    Evaluates the model on the validation set.

    Args:
        model:     The framework model.
        loader:    Validation DataLoader.
        criterion: Loss function.
        device:    Computation device.
        metrics:   EmotionMetrics accumulator.

    Returns:
        Validation metrics dict.
    """
    model.eval()
    metrics.reset()

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        output = model.forward_frame(images)
        loss   = criterion(output['logits'], labels)

        metrics.update(output['probs'], labels, loss=loss.item())

    return metrics.compute()


# ---------------------------------------------------------------------------
# Early stopping helper
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stops training if validation loss doesn't improve for `patience` epochs."""

    def __init__(self, patience: int = 8, mode: str = 'max'):
        self.patience    = patience
        self.mode        = mode
        self.best_value  = float('-inf') if mode == 'max' else float('inf')
        self.counter     = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """
        Updates early stopping state.

        Returns:
            True if training should stop.
        """
        improved = (
            (self.mode == 'max' and value > self.best_value) or
            (self.mode == 'min' and value < self.best_value)
        )

        if improved:
            self.best_value = value
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: dict, args):
    """
    Full training pipeline for the Swin-FANE framework.

    Args:
        config: Configuration dictionary loaded from YAML.
        args:   Parsed command-line arguments.
    """
    # ---- Device setup ----
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"[Train] Using device: {device}")

    # ---- Multiple seeds (paper Section 5.1: 3 independent runs) ----
    n_runs = config['training'].get('num_runs', 3) if args.multi_run else 1

    all_run_results = []

    for run_idx in range(n_runs):
        seed = config['training'].get('seed', 42) + run_idx
        set_seed(seed)
        print(f"\n{'='*60}")
        print(f"  Run {run_idx + 1}/{n_runs}  |  Seed: {seed}")
        print(f"{'='*60}")

        # ---- Logger ----
        logger = TrainingLogger(
            log_dir=config['paths']['log_dir'],
            experiment_name=f"swin_fane_run{run_idx + 1}",
            use_tensorboard=config['logging']['use_tensorboard'],
            use_wandb=config['logging']['use_wandb'],
        )

        # ---- Data loaders ----
        dataset_name = args.dataset or config['dataset']['name']
        logger.info(f"Loading dataset: {dataset_name}")

        if dataset_name.upper() == 'FER2013':
            train_loader, val_loader, _ = build_fer2013_dataloaders(
                root_dir=config['dataset']['root'],
                image_size=config['dataset']['image_size'],
                batch_size=config['training']['batch_size'],
                num_workers=config['training']['num_workers'],
            )
        else:
            train_loader, val_loader, _ = build_fane_dataloaders(
                root_dir=config['dataset']['root'],
                image_size=config['dataset']['image_size'],
                sequence_length=1,                         # Frame-level training
                batch_size=config['training']['batch_size'],
                num_workers=config['training']['num_workers'],
                subject_independent=config['dataset']['subject_independent'],
                seed=seed,
                split_file=os.path.join(config['paths']['results_dir'],
                                        'data_split.json'),
            )

        logger.info(f"Train batches: {len(train_loader)} | "
                    f"Val batches:   {len(val_loader)}")

        # ---- Model ----
        model = build_framework(config).to(device)
        logger.info(f"Model parameters: {model.get_param_count()}")

        # ---- Optionally freeze backbone for first N epochs ----
        if config['training'].get('freeze_backbone_epochs', 0) > 0:
            model.freeze_backbone()
            logger.info("Backbone frozen for first training stage.")

        # ---- Loss function: cross-entropy with label smoothing ----
        criterion = nn.CrossEntropyLoss(
            label_smoothing=config['training'].get('label_smoothing', 0.1)
        )

        # ---- Optimiser (AdamW, paper Section 5.1) ----
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=tuple(config['training']['betas']),
        )

        # ---- LR Scheduler ----
        scheduler_name = config['training'].get('scheduler', 'StepLR')
        if scheduler_name == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config['training']['num_epochs'],
            )
        else:
            scheduler = StepLR(
                optimizer,
                step_size=config['training']['step_size'],
                gamma=config['training']['gamma'],
            )

        # ---- Resume from checkpoint ----
        start_epoch = 1
        if args.resume and os.path.exists(args.resume):
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch'] + 1
            logger.info(f"Resumed from checkpoint: {args.resume} "
                        f"(epoch {ckpt['epoch']})")

        # ---- Metrics + early stopping ----
        train_metrics_acc = EmotionMetrics(num_classes=config['dataset']['num_classes'])
        val_metrics_acc   = EmotionMetrics(num_classes=config['dataset']['num_classes'])
        early_stopping    = EarlyStopping(
            patience=config['training']['early_stopping_patience'],
            mode='max',
        )

        # ---- Training loop ----
        num_epochs = config['training']['num_epochs']
        logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(start_epoch, num_epochs + 1):
            t0 = time.time()

            # Unfreeze backbone after N epochs if it was frozen
            freeze_epochs = config['training'].get('freeze_backbone_epochs', 0)
            if freeze_epochs > 0 and epoch == freeze_epochs + 1:
                model.unfreeze_spatial_encoder()
                logger.info(f"Epoch {epoch}: Spatial encoder UNFROZEN.")

            # Train one epoch
            train_mets = train_epoch(
                model, train_loader, criterion, optimizer, device,
                train_metrics_acc, epoch,
                log_interval=config['logging']['log_interval'],
                logger=logger,
            )

            # Validate
            val_mets = validate_epoch(
                model, val_loader, criterion, device, val_metrics_acc
            )

            # LR step
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Log epoch results
            logger.log_epoch(epoch, train_mets, val_mets, lr=current_lr)

            # Save best checkpoint
            is_best = val_mets['accuracy'] >= logger.best_val_acc
            logger.save_checkpoint(
                model, optimizer, epoch, val_mets,
                checkpoint_dir=config['paths']['checkpoint_dir'],
                is_best=is_best,
            )

            # Early stopping check
            if early_stopping.step(val_mets['accuracy']):
                logger.info(f"Early stopping triggered at epoch {epoch}. "
                            f"Best val acc: {early_stopping.best_value:.2f}%")
                break

            elapsed = time.time() - t0
            logger.info(f"  Epoch {epoch} completed in {elapsed:.1f}s")

        # ---- Record run results ----
        all_run_results.append({
            'run':          run_idx + 1,
            'seed':         seed,
            'best_val_acc': logger.best_val_acc,
            'best_epoch':   logger.best_epoch,
        })
        logger.close()

    # ---- Multi-run summary ----
    if n_runs > 1:
        accs  = [r['best_val_acc'] for r in all_run_results]
        mean  = np.mean(accs)
        std   = np.std(accs)
        print(f"\n{'='*60}")
        print(f"Multi-run summary ({n_runs} runs):")
        print(f"  Val accuracy: {mean:.2f}% ± {std:.2f}%")
        for r in all_run_results:
            print(f"  Run {r['run']} (seed={r['seed']}): "
                  f"{r['best_val_acc']:.2f}% @ epoch {r['best_epoch']}")
        print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Swin-FANE Emotion Recognition Framework'
    )
    parser.add_argument('--config',    type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--resume',    type=str, default=None,
                        help='Path to checkpoint for resuming training')
    parser.add_argument('--dataset',   type=str, default=None,
                        choices=['FANE', 'FER2013'],
                        help='Override dataset choice from config')
    parser.add_argument('--multi-run', action='store_true',
                        help='Run N independent seeds (paper: 3 runs)')
    parser.add_argument('--gpu',       type=int, default=0,
                        help='GPU index to use (default: 0)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # ---- Set CUDA device ----
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # ---- Load config ----
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ---- Create output directories ----
    for dir_key in ['checkpoint_dir', 'results_dir', 'log_dir']:
        os.makedirs(config['paths'][dir_key], exist_ok=True)

    train(config, args)
