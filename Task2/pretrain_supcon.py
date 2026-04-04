"""
Task 2 — Approach B (Step 1): Supervised Contrastive (SupCon) Pre-training

Replaces SimCLR with SupCon (Khosla et al., 2020) which uses class labels
to define positive pairs — all same-class patches in a batch are pulled together,
all different-class patches are pushed apart.

Why SupCon over SimCLR:
  - SimCLR only uses augmented views of the same image as positives (instance-level)
    → encoder learns to be invariant to augmentations but not class-discriminative
  - SupCon uses ALL same-class patches as positives (class-level)
    → encoder explicitly learns features that separate classes
  - Critical for histopathology where inter-class differences are subtle

Each image gets two augmented views. For each anchor in the 2N-sample batch,
positives = all other samples (including the second view) with the same class label.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pathlib import Path

DATA_DIR   = Path(__file__).parent / "task2_dataset"
OUTPUT_DIR = Path(__file__).parent / "checkpoints_b"
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE     = 256
EPOCHS         = 100
LR             = 3e-4
TEMPERATURE    = 0.07   # SupCon uses lower temperature than SimCLR (0.5)
PROJ_DIM       = 128
EARLY_STOP_PAT = 10

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()          else
    torch.device("cpu")
)
print(f"Using device: {DEVICE}")

# Same strong augmentations as SimCLR — two views per image
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
    """Returns two augmented views + class label for each patch."""
    def __init__(self):
        self.X = np.load(DATA_DIR / "X_contrastive.npy")
        self.y = np.load(DATA_DIR / "y_contrastive.npy")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img   = self.X[idx]
        label = int(self.y[idx])
        return supcon_transform(img), supcon_transform(img), label


class SupConEncoder(nn.Module):
    def __init__(self, proj_dim=PROJ_DIM):
        super().__init__()
        backbone     = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # remove fc
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x).flatten(1)   # (B, 512)
        z = self.projector(h)             # (B, proj_dim)
        return F.normalize(z, dim=1)


def supcon_loss(features, labels, temperature=TEMPERATURE):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).

    features: (N, D) L2-normalised projection vectors
    labels:   (N,)  class indices

    For each anchor i, positives P(i) = all j != i with same class label.
    Loss = -1/|P(i)| * sum_{p in P(i)} log(
             exp(f_i · f_p / τ) / sum_{a != i} exp(f_i · f_a / τ)
           )
    """
    N      = features.size(0)
    device = features.device

    sim = torch.mm(features, features.T) / temperature  # (N, N)

    # Positive mask: same class, excluding self
    labels  = labels.view(-1, 1)
    pos_mask = torch.eq(labels, labels.T).float().to(device)
    eye      = torch.eye(N, device=device)
    pos_mask = pos_mask - eye   # remove self-similarity

    # Denominator: all pairs excluding self
    neg_mask  = 1 - eye

    # For numerical stability subtract row max
    sim_max, _ = torch.max(sim * neg_mask, dim=1, keepdim=True)
    sim        = sim - sim_max.detach()

    exp_sim  = torch.exp(sim) * neg_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Only average over positives; skip anchors with no positives
    n_pos = pos_mask.sum(1)
    loss  = -(pos_mask * log_prob).sum(1) / (n_pos + 1e-8)
    loss  = loss[n_pos > 0].mean()   # skip anchors with no positives in batch
    return loss


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for view1, view2, labels in loader:
        # Concatenate both views: (2B, C, H, W) with repeated labels (2B,)
        imgs   = torch.cat([view1, view2], dim=0).to(DEVICE)
        labels = torch.cat([labels, labels], dim=0).to(DEVICE)

        feats = model(imgs)
        loss  = supcon_loss(feats, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * view1.size(0)
    return total_loss / len(loader.dataset)


def main():
    loader = DataLoader(
        SupConDataset(),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        pin_memory=True, drop_last=True
    )
    print(f"Contrastive set: {len(loader.dataset)} patches, "
          f"{len(loader)} batches/epoch")

    model     = SupConEncoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss       = float("inf")
    epochs_no_impro = 0
    loss_history    = []

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, loader, optimizer)
        scheduler.step()
        loss_history.append(loss)
        print(f"Epoch {epoch:03d}/{EPOCHS}  loss={loss:.4f}")

        if loss < best_loss:
            best_loss       = loss
            epochs_no_impro = 0
            torch.save(model.encoder.state_dict(), OUTPUT_DIR / "supcon_encoder.pth")
            print(f"  -> Saved encoder (loss={best_loss:.4f})")
        else:
            epochs_no_impro += 1
            if epochs_no_impro >= EARLY_STOP_PAT:
                print(f"  -> Early stopping at epoch {epoch}")
                break

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.title("SupCon Pre-training Loss"); plt.xlabel("Epoch"); plt.ylabel("SupCon Loss")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "supcon_pretrain_loss.png", dpi=150)
    plt.close()

    print(f"\nPre-training complete. Encoder saved to {OUTPUT_DIR / 'supcon_encoder.pth'}")


if __name__ == "__main__":
    main()
