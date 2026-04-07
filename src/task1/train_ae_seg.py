import argparse
import os
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config
from task1.dataset import get_segmentation_loaders, TissueSegmentationDataset
from task1.models import AEEncoder, AESegmentationModel
from task1.utils import (
    set_seed, get_device,
    SegmentationMetrics, get_criterion, compute_class_weights,
    save_training_curves, save_prediction_grid,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train segmentation decoder on frozen AE encoder")
    p.add_argument("--ae_checkpoint", type=str,
                    default=os.path.join(config.CHECKPOINT_DIR,
                                          "autoencoder_best.pth"),
                    help="Path to pre‑trained autoencoder checkpoint")
    p.add_argument("--epochs", type=int, default=config.AE_SEG_EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.AE_SEG_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.AE_SEG_LR)
    p.add_argument("--weight_decay", type=float,
                    default=config.AE_SEG_WEIGHT_DECAY)
    p.add_argument("--patch_size", type=int, default=config.PATCH_SIZE)
    p.add_argument("--features", type=int, nargs="+",
                    default=config.AE_FEATURES)
    p.add_argument("--data_root", type=str, default=config.DATASET_ROOT)
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--lambda_dice", type=float, default=config.LAMBDA_DICE)
    p.add_argument("--lambda_ce", type=float, default=config.LAMBDA_CE)
    p.add_argument("--checkpoint_dir", type=str, default=config.CHECKPOINT_DIR)
    p.add_argument("--results_dir", type=str, default=config.RESULTS_DIR)
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    # Encoder stays in eval mode (frozen)
    model.encoder.eval()

    running_loss = 0.0
    for images, masks in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

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
    print("  AE-Seg Hyperparameters")
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
    print(f"  AE checkpoint   : {args.ae_checkpoint}")
    print(f"  Seed            : {args.seed}")
    print(f"{'=' * 55}\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Load pre‑trained encoder ─────────────
    if not os.path.exists(args.ae_checkpoint):
        raise FileNotFoundError(
            f"Autoencoder checkpoint not found: {args.ae_checkpoint}\n"
            f"Run train_autoencoder.py first.")

    print(f"Loading pre‑trained encoder from: {args.ae_checkpoint}")
    ckpt = torch.load(args.ae_checkpoint, map_location=device,
                       weights_only=False)

    features = ckpt.get("features", args.features)
    encoder = AEEncoder(features=features)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    print(f"  Loaded encoder (features={features})")

    # ── Build segmentation model ─────────────
    model = AESegmentationModel(encoder, freeze_encoder=True,
                                 features=features).to(device)

    total_params = model.count_parameters(trainable_only=False)
    trainable_params = model.count_parameters(trainable_only=True)
    print(f"AE-Seg total parameters:     {total_params:,}")
    print(f"AE-Seg trainable parameters: {trainable_params:,} (decoder only)")

    # ── Data ─────────────────────────────────
    train_loader, val_loader, _ = get_segmentation_loaders(
        batch_size=args.batch_size, patch_size=args.patch_size)

    # ── Class weights ────────────────────────
    ce_weight = None
    if args.use_class_weights:
        print("Computing class weights from training set ...")
        train_ds = TissueSegmentationDataset(
            config.TRAIN_IMAGE_DIR, config.TRAIN_LABEL_DIR,
            patch_size=args.patch_size, is_train=False)
        ce_weight = compute_class_weights(train_ds).to(device)
        print(f"  Class weights: {ce_weight.tolist()}")

    # ── Loss / optimiser / scheduler ─────────
    criterion = get_criterion(weight=ce_weight,
                              lambda_dice=args.lambda_dice,
                              lambda_ce=args.lambda_ce).to(device)
    # Only optimise decoder parameters
    optimizer = AdamW(model.seg_decoder.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=config.LR_MIN)

    # ── Training loop ────────────────────────
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

        if mean_dice > best_dice:
            best_dice = mean_dice
            ckpt_path = os.path.join(args.checkpoint_dir,
                                      "ae_seg_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_dice": best_dice,
                "features": features,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✓ New best model saved (dice={best_dice:.4f})")

    # ── Save curves ──────────────────────────
    save_training_curves(
        train_losses, val_losses, val_dices,
        os.path.join(args.results_dir, "ae_seg_training_curves.png"))

    # ── Final validation report ──────────────
    print("\n" + "=" * 55)
    print("  FINAL VALIDATION RESULTS (AE‑Seg, best model)")
    print("=" * 55)

    ckpt = torch.load(os.path.join(args.checkpoint_dir, "ae_seg_best.pth"),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    _, final_metrics = validate(model, val_loader, criterion, device)
    final_metrics.print_summary()

    # Save example predictions
    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1)
            save_prediction_grid(
                images, masks, preds,
                os.path.join(args.results_dir,
                             "ae_seg_val_predictions.png"))
            break

    print(f"\nDone.  Best val Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()