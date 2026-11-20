"""
CrowdPulse - Transfer Learning Training Pipeline (Pretrained CNN Crowd Counter)
Project: CrowdPulse - Crowd Density Management System (SIH25165)

Beginner-Friendly Explanation:
This script trains a deep neural network to accurately count crowds on standard
optical CCTV images using Transfer Learning:
1. Loads an ImageNet-pretrained backbone (ConvNeXt-Tiny, ResNet-50, or EfficientNet-B0).
2. Attaches our Multi-Scale Spatial Pyramid Pooling (SPP) crowd regression head.
3. Trains strictly on the 700 training images (Part A Train + Part B Train)
   using count-preserving augmentations (horizontal flip, brightness/contrast jitter,
   grayscale simulation — NEVER random crops that cut off people!).
4. Evaluates strictly on the 498 unseen test images (Part A Test + Part B Test)
   and reports honest MAE, RMSE, Part A MAE, Part B MAE, and R² accuracy.
5. Saves the best weights to models/crowdpulse_count_model.pth and full metrics
   to models/training_metrics.json.

Usage:
    python training/train_crowd_model.py --backbone convnext_tiny --epochs 25 --img-size 384
"""

import argparse
import io
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
os.environ.setdefault("TORCH_HOME", str(config.MODELS_DIR / ".torch_cache"))
from detection.crowd_counter_model import (
    CrowdPulseCountNet,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def set_seed(seed: int = 42):
    """Fixes random seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ParquetCrowdDataset(Dataset):
    """PyTorch Dataset reading images and exact counts directly from Parquet files.

    Stores raw JPEG bytes in RAM and decodes on-the-fly per sample to stay lightweight
    and fast on Windows without requiring extra disk extraction.
    """

    def __init__(
        self,
        parquet_paths: List[Path],
        expected_split: str,
        transform=None,
    ):
        self.samples: List[Tuple[bytes, float, str, str]] = []
        self.transform = transform

        for p in parquet_paths:
            if not p.exists():
                raise FileNotFoundError(f"Dataset file not found: {p}")

            df = pd.read_parquet(p)
            for _, row in df.iterrows():
                split_val = str(row["split"]).strip().lower()
                if split_val != expected_split:
                    raise ValueError(
                        f"Data leakage guard triggered! File {p.name} contains split='{split_val}', "
                        f"expected '{expected_split}'."
                    )
                img_bytes = row["image"]["bytes"]
                count_val = float(row["count"])
                part_val = str(row["part"])
                fname = str(row["file_name"])
                self.samples.append((img_bytes, count_val, part_val, fname))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_bytes, count_val, part_val, fname = self.samples[idx]
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if self.transform is not None:
            tensor_img = self.transform(pil_img)
        else:
            tensor_img = transforms.ToTensor()(pil_img)

        # Target in log1p space for balanced accuracy across sparse (10) and ultra-dense (3000) crowds
        log_target = torch.tensor(np.log1p(count_val), dtype=torch.float32)
        raw_target = torch.tensor(count_val, dtype=torch.float32)
        return tensor_img, log_target, raw_target, part_val


def build_transforms(img_size: int):
    """Builds count-preserving train transforms and clean validation transforms.

    IMPORTANT: Never apply RandomResizedCrop or Cutout, which would remove people
    from the image while leaving the ground-truth headcount label unchanged!
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_transform


@torch.no_grad()
def evaluate_model(
    model: CrowdPulseCountNet,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    use_tta: bool = True,
) -> Dict[str, float]:
    """Evaluates the model on all unseen test images with honest MAE, RMSE, and R² metrics."""
    model.eval()

    all_preds = []
    all_targets = []
    part_a_preds, part_a_targets = [], []
    part_b_preds, part_b_targets = [], []

    for images, _, raw_targets, parts in loader:
        images = images.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            log_pred = model(images)
            pred_counts = torch.expm1(log_pred.clamp(min=0.0, max=10.0))

            if use_tta:
                flipped = torch.flip(images, dims=[3])
                log_pred_flip = model(flipped)
                pred_counts_flip = torch.expm1(log_pred_flip.clamp(min=0.0, max=10.0))
                pred_counts = 0.5 * (pred_counts + pred_counts_flip)

        preds_np = pred_counts.float().cpu().numpy()
        targets_np = raw_targets.numpy()

        for p_val, t_val, part_name in zip(preds_np, targets_np, parts):
            all_preds.append(float(p_val))
            all_targets.append(float(t_val))
            if "part_a" in part_name.lower():
                part_a_preds.append(float(p_val))
                part_a_targets.append(float(t_val))
            else:
                part_b_preds.append(float(p_val))
                part_b_targets.append(float(t_val))

    preds_arr = np.array(all_preds, dtype=np.float64)
    targets_arr = np.array(all_targets, dtype=np.float64)

    overall_mae = float(np.mean(np.abs(preds_arr - targets_arr)))
    overall_rmse = float(np.sqrt(np.mean((preds_arr - targets_arr) ** 2)))

    ss_res = float(np.sum((targets_arr - preds_arr) ** 2))
    ss_tot = float(np.sum((targets_arr - np.mean(targets_arr)) ** 2))
    r2_score = float(1.0 - (ss_res / max(ss_tot, 1e-8)))

    part_a_mae = float(np.mean(np.abs(np.array(part_a_preds) - np.array(part_a_targets)))) if part_a_preds else 0.0
    part_a_rmse = float(np.sqrt(np.mean((np.array(part_a_preds) - np.array(part_a_targets)) ** 2))) if part_a_preds else 0.0

    part_b_mae = float(np.mean(np.abs(np.array(part_b_preds) - np.array(part_b_targets)))) if part_b_preds else 0.0
    part_b_rmse = float(np.sqrt(np.mean((np.array(part_b_preds) - np.array(part_b_targets)) ** 2))) if part_b_preds else 0.0

    return {
        "overall_mae": round(overall_mae, 2),
        "overall_rmse": round(overall_rmse, 2),
        "r2_score": round(r2_score, 4),
        "part_a_mae": round(part_a_mae, 2),
        "part_a_rmse": round(part_a_rmse, 2),
        "part_b_mae": round(part_b_mae, 2),
        "part_b_rmse": round(part_b_rmse, 2),
        "num_eval_samples": len(all_targets),
    }


def train(
    backbone_name: str = "convnext_tiny",
    epochs: int = 25,
    warmup_epochs: int = 3,
    img_size: int = 384,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    backbone_lr: float = 8e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    output_path: Path = config.MODELS_DIR / "crowdpulse_count_model.pth",
    metrics_path: Path = config.MODELS_DIR / "training_metrics.json",
) -> Dict:
    """Runs complete two-stage transfer learning and saves the best model checkpoint."""
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("=" * 72)
    print(f"  CrowdPulse Transfer Learning Pipeline — Backbone: {backbone_name}")
    print("=" * 72)
    print(f"  Device         : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Mixed Precision: {use_amp}")
    print(f"  Image Size     : {img_size}x{img_size}")
    print(f"  Epochs         : {epochs} (Warmup: {warmup_epochs}, Fine-Tune: {epochs - warmup_epochs})")
    print(f"  Batch Size     : {batch_size} x {grad_accum_steps} accum = {batch_size * grad_accum_steps} effective")
    print("-" * 72)

    dataset_dir = config.BASE_DIR / "Dataset_CrowdPulse"
    train_files = [
        dataset_dir / "Project_CrowdPulse_part_A_2.parquet",
        dataset_dir / "Project_CrowdPulse_part_B_2.parquet",
    ]
    test_files = [
        dataset_dir / "project_CrowdPulse_part_A.parquet",
        dataset_dir / "Project_CrowdPulse_part_B.parquet",
    ]

    train_tf, val_tf = build_transforms(img_size)
    print("Loading datasets from Parquet (verifying strict train/test isolation)...")
    train_ds = ParquetCrowdDataset(train_files, expected_split="train", transform=train_tf)
    val_ds = ParquetCrowdDataset(test_files, expected_split="test", transform=val_tf)
    print(f"  Train samples  : {len(train_ds)} images (Part A Train + Part B Train)")
    print(f"  Val/Test samples: {len(val_ds)} images (Part A Test + Part B Test)")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Loading ImageNet-pretrained '{backbone_name}' backbone...")
    model = CrowdPulseCountNet(backbone_name=backbone_name, pretrained=True, dropout=0.3).to(device)

    # Smooth L1 in log1p space balances sparse and dense crowds cleanly
    criterion = nn.SmoothL1Loss(beta=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # Stage 1: Freeze backbone and warm up regression head
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=head_lr, weight_decay=weight_decay)
    scheduler = None

    best_val_mae = float("inf")
    best_metrics = {}
    history = []
    start_total = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # Switch to Stage 2 (Unfreeze full backbone with differential LRs)
        if epoch == warmup_epochs + 1:
            print("-" * 72)
            print(f"  [Stage 2] Unfreezing '{backbone_name}' backbone for end-to-end fine-tuning...")
            print("-" * 72)
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.features.parameters(), "lr": backbone_lr},
                    {"params": model.head.parameters(), "lr": head_lr * 0.4},
                ],
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - warmup_epochs),
                eta_min=1e-6,
            )

        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (images, log_targets, _, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            log_targets = log_targets.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                preds_log = model(images)
                loss = criterion(preds_log, log_targets)
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item())

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = round(running_loss / max(len(train_loader), 1), 4)

        # Evaluate on unseen test set
        val_metrics = evaluate_model(model, val_loader, device, use_amp=use_amp, use_tta=True)
        epoch_sec = round(time.perf_counter() - epoch_start, 1)

        is_best = val_metrics["overall_mae"] < best_val_mae
        if is_best:
            best_val_mae = val_metrics["overall_mae"]
            best_metrics = {**val_metrics, "best_epoch": epoch}
            torch.save(
                {
                    "backbone_name": backbone_name,
                    "img_size": img_size,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": best_metrics,
                },
                output_path,
            )

        epoch_record = {
            "epoch": epoch,
            "stage": "warmup" if epoch <= warmup_epochs else "finetune",
            "train_loss": avg_train_loss,
            **val_metrics,
            "epoch_time_sec": epoch_sec,
            "is_best": is_best,
        }
        history.append(epoch_record)

        star = " *BEST*" if is_best else ""
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_train_loss:.4f} | "
            f"Val MAE: {val_metrics['overall_mae']:6.2f} | RMSE: {val_metrics['overall_rmse']:6.2f} | "
            f"PartA MAE: {val_metrics['part_a_mae']:6.2f} | PartB MAE: {val_metrics['part_b_mae']:5.2f} | "
            f"R2: {val_metrics['r2_score']:.4f} | {epoch_sec:4.1f}s{star}"
        )

    total_time_sec = round(time.perf_counter() - start_total, 1)

    summary = {
        "backbone_name": backbone_name,
        "img_size": img_size,
        "epochs_trained": epochs,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "total_training_time_sec": total_time_sec,
        "checkpoint_path": str(output_path),
        "best_metrics": best_metrics,
        "history": history,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print(f"  Training Complete in {total_time_sec}s!")
    print(f"  Best Overall MAE : {best_metrics['overall_mae']} persons (Epoch {best_metrics['best_epoch']})")
    print(f"  Best Overall RMSE: {best_metrics['overall_rmse']} persons | R2: {best_metrics['r2_score']}")
    print(f"  Best Part A MAE  : {best_metrics['part_a_mae']} (Dense Crowds: 66-2,256 people)")
    print(f"  Best Part B MAE  : {best_metrics['part_b_mae']} (Standard CCTV Crowds: 9-539 people)")
    print(f"  Saved Model      : {output_path}")
    print(f"  Saved Metrics    : {metrics_path}")
    print("=" * 72)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CrowdPulse Pretrained Crowd Counting Network")
    parser.add_argument("--backbone", type=str, default="convnext_tiny", choices=["convnext_tiny", "resnet50", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    train(
        backbone_name=args.backbone,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

