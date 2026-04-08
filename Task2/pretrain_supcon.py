"""
pretrain_supcon.py
------------------
Approach B — Step 1: SupCon encoder pre-training on the contrastive patch set.

Usage:
  python Task2/pretrain_supcon.py --backbone efficientnet_b0
  python Task2/pretrain_supcon.py --backbone resnet18
  python Task2/pretrain_supcon.py --backbone resnet50
"""
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import config
from transforms import supcon_transforms
from patches import SupConDataset
from encoders import SupConModel, supcon_loss


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for view1, view2, labels in loader:
        imgs = torch.cat([view1, view2]).to(config.DEVICE)
        labels = torch.cat([labels, labels]).to(config.DEVICE)
        loss = supcon_loss(model(imgs), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * view1.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="SupCon encoder pre-training")
    parser.add_argument("--backbone", default="efficientnet_b0",
                        choices=["resnet18", "resnet50", "efficientnet_b0"])
    args = parser.parse_args()

    out_dir = config.ROOT_DIR / f"checkpoints_b_{args.backbone}"
    out_dir.mkdir(exist_ok=True)
    print(f"Backbone : {args.backbone}")
    print(f"Device   : {config.DEVICE}")
    print(f"Output   : {out_dir}")

    loader = DataLoader(
        SupConDataset(transform=supcon_transforms),
        batch_size=config.SC_BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    print(f"Contrastive set: {len(loader.dataset):,} patches, "
          f"{len(loader)} batches/epoch")

    model = SupConModel(args.backbone).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.SC_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.SC_EPOCHS)

    best_loss, no_improve, history = float("inf"), 0, []

    for epoch in range(1, config.SC_EPOCHS + 1):
        loss = train_one_epoch(model, loader, optimizer)
        scheduler.step()
        history.append(loss)
        print(f"Epoch {epoch:03d}/{config.SC_EPOCHS}  loss={loss:.4f}")

        if loss < best_loss:
            best_loss, no_improve = loss, 0
            torch.save(model.encoder.state_dict(),
                       out_dir / "supcon_encoder.pth")
            print(f"  -> Saved encoder (loss={best_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= config.SC_EARLY_STOP_PAT:
                print(f"  -> Early stopping at epoch {epoch}")
                break

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(history) + 1), history)
    plt.title(f"SupCon Pre-training Loss — {args.backbone}")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(out_dir / "supcon_pretrain_loss.png", dpi=150)
    plt.close()
    print(f"\nDone. Encoder saved to {out_dir / 'supcon_encoder.pth'}")


if __name__ == "__main__":
    main()
