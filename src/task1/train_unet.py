"""
train_unet.py
-------------
Train the UNet segmentation model on the Puma tissue dataset.

Usage
-----
    python train_unet.py [--epochs 50] [--batch_size 8] [--lr 1e-4]
                         [--patch_size 256] [--features 64 128 256 512 1024]
                         [--data_root ./data]

The script:
  1. Loads train / val splits.
  2. Optionally computes class weights to handle imbalance.
  3. Trains with combined CE + Dice loss.
  4. Saves the best model (by val Dice) and training curves.
"""

import argparse
import os
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config
from task1.dataset import get_segmentation_loaders, TissueSegmentationDataset
from task1.models import UNet
from task1.utils import (
    set_seed, get_device,
    SegmentationMetrics, get_criterion, compute_class_weights,
    save_training_curves, save_prediction_grid,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train UNet for tissue segmentation")
    p.add_argument("--epochs", type=int, default=config.UNET_EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.UNET_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.UNET_LR)
    p.add_argument("--weight_decay", type=float, default=config.UNET_WEIGHT_DECAY)
    p.add_argument("--patch_size", type=int, default=config.PATCH_SIZE)
    p.add_argument("--features", type=int, nargs="+",
                    default=config.UNET_FEATURES)
    p.add_argument("--data_root", type=str, default=config.DATASET_ROOT)
    p.add_argument("--use_class_weights", action="store_true",
                    help="Compute inverse-frequency class weights for CE")
    p.add_argument("--dampen_weights", action="store_true",
                    help="Use sqrt-dampened weights for softer rebalancing")
    p.add_argument("--lambda_dice", type=float, default=config.LAMBDA_DICE)
    p.add_argument("--lambda_ce", type=float, default=config.LAMBDA_CE)
    p.add_argument("--checkpoint_dir", type=str, default=config.CHECKPOINT_DIR)
    p.add_argument("--results_dir", type=str, default=config.RESULTS_DIR)
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    metrics = SegmentationMetrics(device=device)

    for images, masks in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        running_loss += loss.item() * images.size(0)
        metrics.update(logits, masks)

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, metrics


def main():
    args = parse_args()

    if args.data_root != config.DATASET_ROOT:
        config.DATASET_ROOT = args.data_root
        config.TRAIN_IMAGE_DIR = os.path.join(args.data_root, "train", "image")
        config.TRAIN_LABEL_DIR = os.path.join(args.data_root, "train", "tissue")
        config.VAL_IMAGE_DIR = os.path.join(args.data_root, "validation", "image")
        config.VAL_LABEL_DIR = os.path.join(args.data_root, "validation", "tissue")
        config.TEST_IMAGE_DIR = os.path.join(args.data_root, "test", "image")
        config.TEST_LABEL_DIR = os.path.join(args.data_root, "test", "tissue")

    set_seed(args.seed)
    device = get_device()

    print(f"\n{'=' * 55}")
    print("  UNet Hyperparameters")
    print(f"{'=' * 55}")
    print(f"  Device          : {device}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Learning rate   : {args.lr}")
    print(f"  LR min          : {config.LR_MIN}")
    print(f"  Weight decay    : {args.weight_decay}")
    print(f"  Patch size      : {args.patch_size}")
    print(f"  Features        : {args.features}")
    print(f"  Lambda Dice     : {args.lambda_dice}")
    print(f"  Lambda CE       : {args.lambda_ce}")
    print(f"  Class weights   : {args.use_class_weights}")
    print(f"  Dampen weights  : {args.dampen_weights}")
    print(f"  Seed            : {args.seed}")
    print(f"{'=' * 55}\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Data
    train_loader, val_loader, _ = get_segmentation_loaders(
        batch_size=args.batch_size, patch_size=args.patch_size)

    # Class weights
    ce_weight = None
    if args.use_class_weights:
        print("Computing class weights from training set ...")
        train_ds = TissueSegmentationDataset(
            config.TRAIN_IMAGE_DIR, config.TRAIN_LABEL_DIR,
            patch_size=args.patch_size, is_train=False)
        ce_weight = compute_class_weights(train_ds,
                                          dampen=args.dampen_weights).to(device)
        print(f"Class weights: {ce_weight.tolist()}")

    # Model
    model = UNet(features=args.features).to(device)
    print(f"UNet trainable parameters: {model.count_parameters():,}")

    # Loss / optimiser / scheduler
    criterion = get_criterion(weight=ce_weight,
                              lambda_dice=args.lambda_dice,
                              lambda_ce=args.lambda_ce).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=config.LR_MIN)

    # Training loop
    best_dice = 0.0
    train_losses, val_losses, val_dices = [], [], []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        mean_dice = val_metrics.mean_dice()
        elapsed = time.time() - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(mean_dice)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_dice={mean_dice:.4f}  "
              f"({elapsed:.1f}s)")

        # Save best model
        if mean_dice > best_dice:
            best_dice = mean_dice
            ckpt_path = os.path.join(args.checkpoint_dir, "unet_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "args": vars(args),
            }, ckpt_path)
            print(f"New best model saved (dice={best_dice:.4f})")

    # Save training curves
    save_training_curves(
        train_losses, val_losses, val_dices,
        os.path.join(args.results_dir, "unet_training_curves.png"))

    # Final validation report
    print("\n" + "=" * 55)
    print("  FINAL VALIDATION RESULTS (best model)")
    print("=" * 55)

    # Reload best model
    ckpt = torch.load(os.path.join(args.checkpoint_dir, "unet_best.pth"),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    _, final_metrics = validate(model, val_loader, criterion, device)
    final_metrics.print_summary()

    # Save a few example predictions
    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1)
            save_prediction_grid(
                images, masks, preds,
                os.path.join(args.results_dir, "unet_val_predictions.png"))
            break

    print(f"\nDone.  Best val Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()