"""
Task 2 — Approach B (Step 1): SimCLR Self-Supervised Pre-training

Trains a ResNet-18 encoder using SimCLR on the contrastive set (59k patches, no labels).
Saves the encoder weights for use in train_approach_b.py.

SimCLR overview:
  - Each image gets two randomly augmented views
  - Encoder maps each view to a feature vector
  - Projection head maps features to a lower-dim space for contrastive loss
  - NT-Xent loss: pull positive pairs (two views of same image) together,
    push all other pairs apart
  - Projection head is discarded after pre-training; only encoder is kept
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "task2_dataset"
OUTPUT_DIR = Path(__file__).parent / "checkpoints_b"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE      = 256
EPOCHS          = 100
LR              = 3e-4
TEMPERATURE     = 0.5   # NT-Xent temperature τ
PROJ_DIM        = 128   # output dim of projection head
EARLY_STOP_PAT  = 10

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()          else
    torch.device("cpu")
)
print(f"Using device: {DEVICE}")

# ── SimCLR Augmentations ───────────────────────────────────────────────────────
# Two randomised views of each patch; strong augmentation is key to SimCLR
simclr_transform = transforms.Compose([
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


class ContrastiveDataset(Dataset):
    """Returns two augmented views of each patch (no labels needed)."""
    def __init__(self):
        self.X = np.load(DATA_DIR / "X_contrastive.npy")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        return simclr_transform(img), simclr_transform(img)


# ── Model ──────────────────────────────────────────────────────────────────────
class SimCLR(nn.Module):
    def __init__(self, proj_dim=PROJ_DIM):
        super().__init__()
        backbone   = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # remove fc
        # 2-layer MLP projection head: 512 → 256 → proj_dim
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x).flatten(1)   # (B, 512)
        z = self.projector(h)             # (B, proj_dim)
        return F.normalize(z, dim=1)      # L2-normalise for cosine similarity


# ── NT-Xent Loss ───────────────────────────────────────────────────────────────
def nt_xent_loss(z1, z2, temperature=TEMPERATURE):
    """
    Normalised Temperature-scaled Cross Entropy loss.
    z1, z2: (B, D) L2-normalised projection vectors.
    For each sample i, the positive pair is (z1_i, z2_i);
    all 2(B-1) other samples are negatives.
    """
    B    = z1.size(0)
    z    = torch.cat([z1, z2], dim=0)              # (2B, D)
    sim  = torch.mm(z, z.T) / temperature           # (2B, 2B) cosine sim matrix

    # Mask out self-similarity on the diagonal
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))

    # Positive pair indices: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)

    return F.cross_entropy(sim, labels)


# ── Training ───────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for view1, view2 in loader:
        view1, view2 = view1.to(DEVICE), view2.to(DEVICE)
        z1, z2 = model(view1), model(view2)
        loss   = nt_xent_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * view1.size(0)
    return total_loss / len(loader.dataset)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    loader = DataLoader(
        ContrastiveDataset(),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        pin_memory=True, drop_last=True   # drop_last keeps batch size consistent for loss
    )
    print(f"Contrastive set: {len(loader.dataset)} patches, "
          f"{len(loader)} batches/epoch")

    model     = SimCLR().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss       = float("inf")
    epochs_no_impro = 0

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, loader, optimizer)
        scheduler.step()
        print(f"Epoch {epoch:03d}/{EPOCHS}  loss={loss:.4f}")

        if loss < best_loss:
            best_loss       = loss
            epochs_no_impro = 0
            # Save only the encoder (projection head is discarded after pre-training)
            torch.save(model.encoder.state_dict(), OUTPUT_DIR / "simclr_encoder.pth")
            print(f"  -> Saved encoder (loss={best_loss:.4f})")
        else:
            epochs_no_impro += 1
            if epochs_no_impro >= EARLY_STOP_PAT:
                print(f"  -> Early stopping at epoch {epoch}")
                break

    print(f"\nPre-training complete. Encoder saved to {OUTPUT_DIR / 'simclr_encoder.pth'}")


if __name__ == "__main__":
    main()
