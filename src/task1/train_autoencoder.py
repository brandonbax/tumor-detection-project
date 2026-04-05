import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config
from task1.dataset import get_unlabelled_loaders
from task1.models import Autoencoder
from task1.utils import (
    set_seed, get_device,
    save_reconstruction_grid,
)


def parse_args():
    p = argparse.ArgumentParser(description="Pre‑train autoencoder")
    p.add_argument("--epochs", type=int, default=config.AE_EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.AE_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.AE_LR)
    p.add_argument("--weight_decay", type=float, default=config.AE_WEIGHT_DECAY)
    p.add_argument("--patch_size", type=int, default=config.PATCH_SIZE)
    p.add_argument("--features", type=int, nargs="+",
                    default=[64, 128, 256, 512])
    p.add_argument("--data_root", type=str, default=config.DATASET_ROOT)
    p.add_argument("--checkpoint_dir", type=str, default=config.CHECKPOINT_DIR)
    p.add_argument("--results_dir", type=str, default=config.RESULTS_DIR)
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device, non_blocking=True)

        # Target for reconstruction: un‑normalise to [0, 1]
        # (the decoder outputs sigmoid → [0, 1])
        with torch.no_grad():
            mean = torch.tensor([0.485, 0.456, 0.406],
                                device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225],
                               device=device).view(1, 3, 1, 1)
            target = images * std + mean    # back to [0, 1]
            target = target.clamp(0, 1)

        reconstruction = model(images)
        loss = criterion(reconstruction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for images in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        target = (images * std + mean).clamp(0, 1)

        reconstruction = model(images)
        loss = criterion(reconstruction, target)
        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def main():
    args = parse_args()

    if args.data_root != config.DATASET_ROOT:
        config.DATASET_ROOT = args.data_root
        config.TRAIN_IMAGE_DIR = os.path.join(args.data_root, "train", "image")
        config.VAL_IMAGE_DIR = os.path.join(args.data_root, "validation", "image")

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Data ─────────────────────────────────
    train_loader, val_loader = get_unlabelled_loaders(
        batch_size=args.batch_size, patch_size=args.patch_size)

    # ── Model ────────────────────────────────
    model = Autoencoder(features=args.features).to(device)
    print(f"Autoencoder trainable parameters: {model.count_parameters():,}")

    # ── Loss / optimiser / scheduler ─────────
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ────────────────────────
    best_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_mse={train_loss:.6f}  "
              f"val_mse={val_loss:.6f}  "
              f"({elapsed:.1f}s)")

        if val_loss < best_loss:
            best_loss = val_loss
            ckpt_path = os.path.join(args.checkpoint_dir,
                                      "autoencoder_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "best_loss": best_loss,
                "features": args.features,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✓ New best model saved (mse={best_loss:.6f})")

    # ── Save loss curves ─────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train MSE")
    ax.plot(range(1, len(val_losses) + 1), val_losses, label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Pre-training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = os.path.join(args.results_dir, "ae_training_curves.png")
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved AE training curves → {curve_path}")

    # ── Save reconstruction examples ─────────
    model.eval()
    ckpt = torch.load(os.path.join(args.checkpoint_dir,
                                    "autoencoder_best.pth"),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    with torch.no_grad():
        for images in val_loader:
            images = images.to(device)
            recons = model(images)
            save_reconstruction_grid(
                images, recons,
                os.path.join(args.results_dir, "ae_reconstructions.png"))
            break

    print(f"\nDone.  Best val MSE: {best_loss:.6f}")


if __name__ == "__main__":
    main()