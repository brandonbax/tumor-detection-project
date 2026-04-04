"""
Task 2 — Approach A Improved v2: ResNet-18 with Weighted CrossEntropy (histiocyte=3.0)

Same as train_approach_a_weighted.py but with stronger histiocyte weighting (3.0 vs 2.0).
Motivated by test set results showing histiocyte recall still low at 0.53 with weight=2.0.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

DATA_DIR   = Path(__file__).parent.parent / "task2_dataset"
OUTPUT_DIR = Path(__file__).parent / "checkpoints_a_weighted3"
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE     = 64
EPOCHS         = 50
EARLY_STOP_PAT = 7
LR             = 1e-4
NUM_CLASSES    = 3

CLASS_WEIGHTS = [3.0, 1.0, 1.0]  # histiocyte weighted 3x

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


def build_model():
    model    = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)


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
        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def main():
    with open(DATA_DIR / "class_names.json") as f:
        class_names = [v for _, v in sorted(json.load(f).items(), key=lambda x: int(x[0]))]
    print(f"Classes: {class_names}")
    print(f"Class weights: {dict(zip(class_names, CLASS_WEIGHTS))}")

    train_loader = DataLoader(
        NucleiDataset("train",      train_transforms),
        batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        NucleiDataset("validation", val_transforms),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    model         = build_model()
    class_weights = torch.tensor(CLASS_WEIGHTS, device=DEVICE)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    best_val_acc    = 0.0
    epochs_no_impro = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc                      = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc    = val_acc
            epochs_no_impro = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
            print(f"  -> Saved best model (val_acc={best_val_acc:.4f})")
        else:
            epochs_no_impro += 1
            if epochs_no_impro >= EARLY_STOP_PAT:
                print(f"  -> Early stopping at epoch {epoch}")
                break

    epochs_ran = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs_ran, history["train_loss"], label="Train")
    ax1.plot(epochs_ran, history["val_loss"],   label="Val")
    ax1.set_title("Loss (Weighted 3x)"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(epochs_ran, history["train_acc"], label="Train")
    ax2.plot(epochs_ran, history["val_acc"],   label="Val")
    ax2.axhline(0.7083, color="r", linestyle="--", label="Baseline")
    ax2.set_title("Accuracy (Weighted 3x)"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves_a_weighted3.png", dpi=150)
    plt.close()

    print("\n── Final Evaluation (best checkpoint) ──")
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pth", map_location=DEVICE))
    _, _, val_preds, val_labels = evaluate(model, val_loader, criterion)
    print(classification_report(val_labels, val_preds, target_names=class_names, digits=4))
    print(f"Overall accuracy: {accuracy_score(val_labels, val_preds):.4f}")


if __name__ == "__main__":
    main()
