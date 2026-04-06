"""
Task 2 — Approach B (Step 2): Encoder + Linear Classifier

Usage:
  python train_approach_b.py --backbone efficientnet_b0
  python train_approach_b.py --backbone efficientnet_b0 --unfreeze   # unfreeze last block
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, accuracy_score, silhouette_score
from sklearn.manifold import TSNE
from pathlib import Path

DATA_DIR = Path(__file__).parent / "task2_dataset"

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


def load_encoder(backbone, ckpt_path, unfreeze=False):
    if backbone == "resnet18":
        m       = models.resnet18(weights=None)
        encoder = nn.Sequential(*list(m.children())[:-1])
        feat_dim = 512
        # last block = layer4 (index 7)
        unfreeze_modules = [list(encoder.children())[7]] if unfreeze else []
    elif backbone == "resnet50":
        m       = models.resnet50(weights=None)
        encoder = nn.Sequential(*list(m.children())[:-1])
        feat_dim = 2048
        unfreeze_modules = [list(encoder.children())[7]] if unfreeze else []
    elif backbone == "efficientnet_b0":
        m       = models.efficientnet_b0(weights=None)
        encoder = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1))
        feat_dim = 1280
        # features[7] = last MBConv block, features[8] = head conv (1x1 → 1280)
        unfreeze_modules = [m.features[7], m.features[8]] if unfreeze else []

    encoder.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    # freeze everything first
    for p in encoder.parameters():
        p.requires_grad = False
    # then selectively unfreeze
    for mod in unfreeze_modules:
        for p in mod.parameters():
            p.requires_grad = True

    n_frozen   = sum(1 for p in encoder.parameters() if not p.requires_grad)
    n_unfrozen = sum(1 for p in encoder.parameters() if p.requires_grad)
    print(f"  Encoder params — frozen: {n_frozen}, unfrozen: {n_unfrozen}")
    return encoder.to(DEVICE), feat_dim


class EncoderClassifier(nn.Module):
    def __init__(self, encoder, feat_dim, num_classes):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        f = self.encoder(x).flatten(1)
        return self.classifier(f)


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


@torch.no_grad()
def extract_features(encoder, loader):
    encoder.eval()
    feats, labels = [], []
    for imgs, lbls in loader:
        f = encoder(imgs.to(DEVICE)).flatten(1).cpu().numpy()
        feats.append(f)
        labels.extend(lbls.numpy())
    return np.concatenate(feats), np.array(labels)


def plot_tsne(features, labels, class_names, title, save_path):
    print("Running t-SNE...")
    reduced = TSNE(n_components=2, perplexity=40, random_state=42,
                   max_iter=1000).fit_transform(features)
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        mask = labels == i
        plt.scatter(reduced[mask, 0], reduced[mask, 1],
                    label=name, alpha=0.5, s=8)
    plt.legend(markerscale=2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  t-SNE saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet18",
                        choices=["resnet18", "resnet50", "efficientnet_b0"])
    parser.add_argument("--unfreeze", action="store_true",
                        help="Unfreeze last encoder block + head for fine-tuning")
    args = parser.parse_args()

    run_suffix = "_unfrozen" if args.unfreeze else ""
    ckpt_dir   = Path(__file__).parent / f"checkpoints_b_{args.backbone}{run_suffix}"
    ckpt_dir.mkdir(exist_ok=True)
    encoder_ckpt = Path(__file__).parent / f"checkpoints_b_{args.backbone}" / "supcon_encoder.pth"
    print(f"Backbone: {args.backbone} | Unfreeze last block: {args.unfreeze}")
    print(f"Encoder checkpoint: {encoder_ckpt}")
    print(f"Output dir: {ckpt_dir}")
    print(f"Using device: {DEVICE}")

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

    encoder, feat_dim = load_encoder(args.backbone, encoder_ckpt, unfreeze=args.unfreeze)
    model     = EncoderClassifier(encoder, feat_dim, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if args.unfreeze:
        # differential LRs: unfrozen encoder layers get 10x lower LR to avoid destroying SupCon features
        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam([
            {"params": encoder_params,           "lr": LR * 0.01},
            {"params": model.classifier.parameters(), "lr": LR},
        ])
    else:
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    best_val_acc, epochs_no_impro = 0.0, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc                        = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc, val_preds, val_labels   = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, epochs_no_impro = val_acc, 0
            torch.save(model.state_dict(), ckpt_dir / "best_model_b.pth")
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
    ax1.set_title(f"Loss — {args.backbone}"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(epochs_ran, history["train_acc"], label="Train")
    ax2.plot(epochs_ran, history["val_acc"],   label="Val")
    ax2.axhline(0.7083, color="r", linestyle="--", label="Baseline")
    ax2.set_title(f"Accuracy — {args.backbone}"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout()
    plt.savefig(ckpt_dir / f"training_curves_b_{args.backbone}.png", dpi=150)
    plt.close()

    print("\n── Final Evaluation ──")
    model.load_state_dict(torch.load(ckpt_dir / "best_model_b.pth", map_location=DEVICE))
    _, _, val_preds, val_labels = evaluate(model, val_loader, criterion)
    print(classification_report(val_labels, val_preds, target_names=class_names, digits=4))
    print(f"Overall accuracy: {accuracy_score(val_labels, val_preds):.4f}")

    print("\n── Latent Space Evaluation ──")
    feats, lbls = extract_features(encoder, val_loader)
    sil = silhouette_score(feats, lbls, sample_size=2000, random_state=42)
    print(f"Silhouette score: {sil:.4f}")
    mode_label = "unfrozen last block" if args.unfreeze else "frozen encoder"
    plot_tsne(feats, lbls, class_names,
              f"t-SNE — SupCon {args.backbone} ({mode_label})",
              ckpt_dir / f"tsne_{args.backbone}.png")


if __name__ == "__main__":
    main()
