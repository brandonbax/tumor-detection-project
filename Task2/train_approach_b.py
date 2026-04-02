"""
Task 2 — Approach B (Step 2): Frozen Encoder + Linear Classifier

Loads the SimCLR pre-trained encoder, freezes it, and trains a linear
classification head on the 7500 labelled training patches.

Also evaluates the latent space quality using:
  - t-SNE visualisation
  - Silhouette score
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent / "task2_dataset"
CKPT_DIR    = Path(__file__).parent / "checkpoints_b"
OUTPUT_DIR  = Path(__file__).parent / "checkpoints_b"

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE     = 64
EPOCHS         = 50
LR             = 1e-3
EARLY_STOP_PAT = 7
NUM_CLASSES    = 3

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()          else
    torch.device("cpu")
)
print(f"Using device: {DEVICE}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class NucleiDataset(Dataset):
    def __init__(self, split, transform=None):
        self.X         = np.load(DATA_DIR / f"X_{split}.npy")
        self.y         = np.load(DATA_DIR / f"y_{split}.npy")
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img   = self.X[idx]
        label = int(self.y[idx])
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Model ──────────────────────────────────────────────────────────────────────
class FrozenEncoderClassifier(nn.Module):
    def __init__(self, encoder, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x).flatten(1)  # (B, 512) — frozen
        return self.classifier(features)


def load_encoder():
    backbone = models.resnet18(weights=None)
    encoder  = nn.Sequential(*list(backbone.children())[:-1])
    encoder.load_state_dict(torch.load(CKPT_DIR / "simclr_encoder.pth", map_location=DEVICE))
    for param in encoder.parameters():
        param.requires_grad = False   # freeze encoder
    return encoder.to(DEVICE)


# ── Training ───────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs     = model(imgs)
        loss        = criterion(outputs, labels)
        preds       = outputs.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ── Latent Space Evaluation ────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(encoder, loader):
    encoder.eval()
    feats, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(DEVICE)
        f    = encoder(imgs).flatten(1).cpu().numpy()
        feats.append(f)
        labels.extend(lbls.numpy())
    return np.concatenate(feats), np.array(labels)


def plot_tsne(features, labels, class_names, save_path):
    print("Running t-SNE (this may take ~1 min)...")
    tsne    = TSNE(n_components=2, perplexity=40, random_state=42, n_iter=1000)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    colors = ["#e6194b", "#4363d8", "#3cb44b"]
    for i, name in enumerate(class_names):
        mask = labels == i
        plt.scatter(reduced[mask, 0], reduced[mask, 1],
                    c=colors[i], label=name, alpha=0.5, s=8)
    plt.legend(markerscale=2)
    plt.title("t-SNE of SimCLR Encoder Features (Validation Set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  t-SNE plot saved to {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    with open(DATA_DIR / "class_names.json") as f:
        class_names = [v for _, v in sorted(json.load(f).items(), key=lambda x: int(x[0]))]
    print(f"Classes: {class_names}")

    train_loader = DataLoader(
        NucleiDataset("train",      train_transforms),
        batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        NucleiDataset("validation", val_transforms),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    encoder   = load_encoder()
    model     = FrozenEncoderClassifier(encoder).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # Higher LR than Approach A — only the linear head has trainable params
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    best_val_loss   = float("inf")
    best_val_acc    = 0.0
    epochs_no_impro = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc                    = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc, val_preds, val_lbls = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            best_val_acc    = val_acc
            epochs_no_impro = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model_b.pth")
            print(f"  -> Saved best model (val_acc={best_val_acc:.4f})")
        else:
            epochs_no_impro += 1
            if epochs_no_impro >= EARLY_STOP_PAT:
                print(f"  -> Early stopping at epoch {epoch}")
                break

    # ── Final evaluation ───────────────────────────────────────────────────────
    print("\n── Final Evaluation (best checkpoint) ──")
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model_b.pth", map_location=DEVICE))
    _, _, val_preds, val_lbls = evaluate(model, val_loader, criterion)
    print(classification_report(val_lbls, val_preds, target_names=class_names, digits=4))
    print(f"Overall accuracy: {accuracy_score(val_lbls, val_preds):.4f}")

    # ── Latent space evaluation ────────────────────────────────────────────────
    print("\n── Latent Space Evaluation ──")
    features, labels = extract_features(encoder, val_loader)

    sil = silhouette_score(features, labels, sample_size=2000, random_state=42)
    print(f"Silhouette score: {sil:.4f}  (range -1 to 1, higher = better separated)")

    plot_tsne(features, labels, class_names, OUTPUT_DIR / "tsne_simclr.png")


if __name__ == "__main__":
    main()
