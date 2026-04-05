"""
Task 2 — Approach B (Step 1): SupCon Pre-training

Usage:
  python pretrain_supcon.py --backbone resnet18
  python pretrain_supcon.py --backbone resnet50
  python pretrain_supcon.py --backbone efficientnet_b0
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pathlib import Path

DATA_DIR = Path(__file__).parent / "task2_dataset"

BATCH_SIZE     = 256
EPOCHS         = 100
LR             = 3e-4
TEMPERATURE    = 0.07
PROJ_DIM       = 128
EARLY_STOP_PAT = 10

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()          else
    torch.device("cpu")
)

supcon_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(size=96, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=9)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class SupConDataset(Dataset):
    def __init__(self):
        self.X = np.load(DATA_DIR / "X_contrastive.npy")
        self.y = np.load(DATA_DIR / "y_contrastive.npy")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx]
        return supcon_transform(img), supcon_transform(img), int(self.y[idx])


def build_encoder(backbone):
    """Returns (encoder, feature_dim)."""
    if backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        encoder = nn.Sequential(*list(m.children())[:-1])
        return encoder, 512

    elif backbone == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        encoder = nn.Sequential(*list(m.children())[:-1])
        return encoder, 2048

    elif backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # features = Sequential of blocks; avgpool handled separately
        encoder = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1))
        return encoder, 1280

    else:
        raise ValueError(f"Unknown backbone: {backbone}")


class SupConModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder, feat_dim = build_encoder(backbone)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, PROJ_DIM),
        )

    def forward(self, x):
        h = self.encoder(x).flatten(1)
        z = self.projector(h)
        return F.normalize(z, dim=1)


def supcon_loss(features, labels, temperature=TEMPERATURE):
    N      = features.size(0)
    device = features.device
    sim    = torch.mm(features, features.T) / temperature

    labels   = labels.view(-1, 1)
    pos_mask = torch.eq(labels, labels.T).float().to(device)
    eye      = torch.eye(N, device=device)
    pos_mask = pos_mask - eye

    neg_mask   = 1 - eye
    sim_max, _ = torch.max(sim * neg_mask, dim=1, keepdim=True)
    sim        = sim - sim_max.detach()

    exp_sim  = torch.exp(sim) * neg_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    n_pos            = pos_mask.sum(1)
    mean_log_prob    = (pos_mask * log_prob).sum(1) / (n_pos + 1e-8)
    return -mean_log_prob[n_pos > 0].mean()


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for view1, view2, labels in loader:
        imgs   = torch.cat([view1, view2], dim=0).to(DEVICE)
        labels = torch.cat([labels, labels], dim=0).to(DEVICE)
        feats  = model(imgs)
        loss   = supcon_loss(feats, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * view1.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet18",
                        choices=["resnet18", "resnet50", "efficientnet_b0"])
    args = parser.parse_args()

    output_dir = Path(__file__).parent / f"checkpoints_b_{args.backbone}"
    output_dir.mkdir(exist_ok=True)
    print(f"Backbone: {args.backbone} | Output: {output_dir}")
    print(f"Using device: {DEVICE}")

    loader = DataLoader(
        SupConDataset(), batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    print(f"Contrastive set: {len(loader.dataset)} patches, {len(loader)} batches/epoch")

    model     = SupConModel(args.backbone).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss, epochs_no_impro, loss_history = float("inf"), 0, []

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, loader, optimizer)
        scheduler.step()
        loss_history.append(loss)
        print(f"Epoch {epoch:03d}/{EPOCHS}  loss={loss:.4f}")

        if loss < best_loss:
            best_loss, epochs_no_impro = loss, 0
            torch.save(model.encoder.state_dict(), output_dir / "supcon_encoder.pth")
            print(f"  -> Saved encoder (loss={best_loss:.4f})")
        else:
            epochs_no_impro += 1
            if epochs_no_impro >= EARLY_STOP_PAT:
                print(f"  -> Early stopping at epoch {epoch}")
                break

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.title(f"SupCon Loss — {args.backbone}")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "supcon_pretrain_loss.png", dpi=150)
    plt.close()
    print(f"Done. Encoder saved to {output_dir / 'supcon_encoder.pth'}")


if __name__ == "__main__":
    main()
