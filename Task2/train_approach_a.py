"""
train_approach_a.py
-------------------
Approach A: fine-tune EfficientNet-B0 end-to-end on the 3-class nuclei task.

EfficientNet-B0 was chosen because its ~5.3M parameters closely match the
baseline budget and its compound scaling gives better accuracy per parameter
than ResNet-18 on small medical datasets.

Usage:
  python Task2/train_approach_a.py
"""
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import classification_report, accuracy_score

import config
from transforms import train_transforms, val_transforms
from patches import get_loaders
from metrics import save_training_curves


def build_model():
    """Load ImageNet-pretrained EfficientNet-B0 and replace the classification head."""
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, config.NUM_CLASSES)
    return m.to(config.DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
        out   = model(imgs)
        loss  = criterion(out, labels)
        preds = out.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def main():
    config.CKPT_A_DIR.mkdir(exist_ok=True)
    print(f"Device: {config.DEVICE}")
    print(f"Class weights: {dict(zip(config.TARGET_CLASSES, config.A_CLASS_WEIGHTS))}")

    train_loader, val_loader = get_loaders(
        config.A_BATCH_SIZE, train_transforms, val_transforms)

    model = build_model()
    weights = torch.tensor(config.A_CLASS_WEIGHTS, device=config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.A_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, factor=0.5)

    best_val_acc, no_improve = 0.0, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, config.A_EPOCHS + 1):
        train_loss, train_acc                      = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        for k, v in zip(history, [train_loss, val_loss, train_acc, val_acc]):
            history[k].append(v)

        print(f"Epoch {epoch:02d}/{config.A_EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, no_improve = val_acc, 0
            torch.save(model.state_dict(), config.CKPT_A_DIR / "best_model.pth")
            print(f"  -> Best model saved (val_acc={best_val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= config.A_EARLY_STOP_PAT:
                print(f"  -> Early stopping at epoch {epoch}"); break

    save_training_curves(history,
                         config.CKPT_A_DIR / "training_curves_a.png",
                         title_prefix="Approach A (EfficientNet-B0)")

    print("\nFinal Evaluation (best checkpoint)")
    model.load_state_dict(
        torch.load(config.CKPT_A_DIR / "best_model.pth", map_location=config.DEVICE))
    _, _, val_preds, val_labels = evaluate(model, val_loader, criterion)
    print(classification_report(val_labels, val_preds,
                                target_names=config.TARGET_CLASSES, digits=4))
    print(f"Overall accuracy: {accuracy_score(val_labels, val_preds):.4f}")


if __name__ == "__main__":
    main()
