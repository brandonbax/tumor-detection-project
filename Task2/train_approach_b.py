"""
train_approach_b.py
-------------------
Approach B — Step 2: train a linear classifier on a SupCon-pretrained encoder.

Three unfreezing strategies are supported:
  (default)      — frozen encoder, only head trained (linear probe)
  --unfreeze     — last MBConv block + head conv unfrozen; differential LR
  --unfreeze_all — full encoder fine-tuned at very low LR (LR × 0.001)

Usage:
  python Task2/train_approach_b.py --backbone efficientnet_b0
  python Task2/train_approach_b.py --backbone efficientnet_b0 --unfreeze
  python Task2/train_approach_b.py --backbone efficientnet_b0 --unfreeze_all
  python Task2/train_approach_b.py --backbone efficientnet_b0 --unfreeze --weighted
"""
import argparse
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, silhouette_score

import config
from transforms import train_transforms, val_transforms
from patches import get_loaders
from encoders import build_encoder, EncoderClassifier
from metrics import save_training_curves, extract_features, plot_tsne


def load_encoder(backbone, unfreeze, unfreeze_all):
    """Load SupCon-pretrained encoder weights and apply freeze strategy."""
    ckpt = config.ROOT_DIR / f"checkpoints_b_{backbone}" / "supcon_encoder.pth"
    encoder, feat_dim = build_encoder(backbone, pretrained=False)
    encoder.load_state_dict(torch.load(ckpt, map_location=config.DEVICE))

    # Start fully frozen, then selectively open layers
    for p in encoder.parameters():
        p.requires_grad = False

    if unfreeze_all:
        for p in encoder.parameters():
            p.requires_grad = True
        n = sum(p.numel() for p in encoder.parameters())
        print(f"  Full encoder unfreeze: {n:,} params")
    elif unfreeze:
        # Unfreeze last MBConv block (features[7]) and head conv (features[8])
        # for EfficientNet-B0, or layer4 for ResNets — most task-specific layers
        mods = ([encoder[0][7], encoder[0][8]] if backbone == "efficientnet_b0"
                else [list(encoder.children())[7]])
        for mod in mods:
            for p in mod.parameters():
                p.requires_grad = True
        n = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"  Partial unfreeze (last block): {n:,} params")

    return encoder.to(config.DEVICE), feat_dim


def build_optimizer(model, unfreeze, unfreeze_all):
    """Differential LR: very low for encoder layers, standard for head."""
    if unfreeze_all:
        return torch.optim.Adam([
            {"params": model.encoder.parameters(),    "lr": config.B_LR * 0.001},
            {"params": model.classifier.parameters(), "lr": config.B_LR},
        ])
    elif unfreeze:
        enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
        return torch.optim.Adam([
            {"params": enc_params,                    "lr": config.B_LR * 0.01},
            {"params": model.classifier.parameters(), "lr": config.B_LR},
        ])
    else:
        return torch.optim.Adam(model.classifier.parameters(), lr=config.B_LR)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
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
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Approach B linear probe / fine-tune")
    parser.add_argument("--backbone", default="efficientnet_b0",
                        choices=["resnet18", "resnet50", "efficientnet_b0"])
    parser.add_argument("--unfreeze",     action="store_true",
                        help="Unfreeze last encoder block with differential LR")
    parser.add_argument("--unfreeze_all", action="store_true",
                        help="Unfreeze entire encoder at LR × 0.001")
    parser.add_argument("--weighted",     action="store_true",
                        help="Weighted CrossEntropy — histiocyte weight=3.0")
    args = parser.parse_args()

    run_suffix = (
        "_unfrozen_all" if args.unfreeze_all else
        "_unfrozen"     if args.unfreeze     else ""
    ) + ("_weighted" if args.weighted else "")

    out_dir = config.ROOT_DIR / f"checkpoints_b_{args.backbone}{run_suffix}"
    out_dir.mkdir(exist_ok=True)
    print(f"Backbone : {args.backbone}{run_suffix}")
    print(f"Device   : {config.DEVICE}")

    train_loader, val_loader = get_loaders(
        config.B_BATCH_SIZE, train_transforms, val_transforms)

    encoder, feat_dim = load_encoder(args.backbone, args.unfreeze, args.unfreeze_all)
    model = EncoderClassifier(encoder, feat_dim, config.NUM_CLASSES).to(config.DEVICE)

    weights   = torch.tensor([3.0, 1.0, 1.0], device=config.DEVICE) if args.weighted else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = build_optimizer(model, args.unfreeze, args.unfreeze_all)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, factor=0.5)

    best_val_acc, no_improve = 0.0, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, config.B_EPOCHS + 1):
        train_loss, train_acc                      = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        for k, v in zip(history, [train_loss, val_loss, train_acc, val_acc]):
            history[k].append(v)

        print(f"Epoch {epoch:02d}/{config.B_EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, no_improve = val_acc, 0
            torch.save(model.state_dict(), out_dir / "best_model_b.pth")
            print(f"  -> Best saved (val_acc={best_val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= config.B_EARLY_STOP_PAT:
                print(f"  -> Early stopping at epoch {epoch}"); break

    save_training_curves(history,
                         out_dir / f"training_curves_b_{args.backbone}.png",
                         title_prefix=f"Approach B ({args.backbone}{run_suffix})")

    print("\nFinal Evaluation")
    model.load_state_dict(
        torch.load(out_dir / "best_model_b.pth", map_location=config.DEVICE))
    _, _, val_preds, val_labels = evaluate(model, val_loader, criterion)
    print(classification_report(val_labels, val_preds,
                                target_names=config.TARGET_CLASSES, digits=4))
    print(f"Overall accuracy: {accuracy_score(val_labels, val_preds):.4f}")

    print("\nLatent Space Evaluation")
    feats, lbls = extract_features(encoder, val_loader)
    sil  = silhouette_score(feats, lbls, sample_size=2000, random_state=42)
    mode = ("unfrozen_all" if args.unfreeze_all else
            "unfrozen"     if args.unfreeze     else "frozen")
    print(f"Silhouette score: {sil:.4f}")
    plot_tsne(feats, lbls, config.TARGET_CLASSES,
              f"t-SNE — SupCon {args.backbone} ({mode})",
              out_dir / f"tsne_{args.backbone}.png")


if __name__ == "__main__":
    main()
