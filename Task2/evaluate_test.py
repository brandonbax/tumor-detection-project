"""
evaluate_test.py
----------------
Evaluate all trained models on the official Task 2 test set (1,858 patches).

Labels are inferred from filenames, e.g.:
  test_set_metastatic_roi_013_nuclei_tumor_<uuid>.npy  →  nuclei_tumor

Usage:
  python Task2/evaluate_test.py
  python Task2/evaluate_test.py --tta   # average over 4 flipped views
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score

import config
from transforms import val_transforms
from patches import TestSetDataset
from encoders import build_encoder, EncoderClassifier
from metrics import plot_confusion_matrix, metrics_dict

OUTPUT_DIR = config.ROOT_DIR / "test_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_model_a():
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, config.NUM_CLASSES)
    m.load_state_dict(torch.load(config.CKPT_A_DIR / "best_model.pth",
                                 map_location=config.DEVICE))
    return m.to(config.DEVICE).eval()


def load_model_b(variant: str):
    """Load EncoderClassifier from checkpoints_b_<variant>/best_model_b.pth."""
    ckpt = config.ROOT_DIR / f"checkpoints_b_{variant}" / "best_model_b.pth"
    # Strip variant suffixes to find the base backbone architecture
    base = variant.replace("_unfrozen_all", "").replace("_unfrozen", "").replace("_weighted", "")
    encoder, feat_dim = build_encoder(base, pretrained=False)
    model = EncoderClassifier(encoder, feat_dim, config.NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt, map_location=config.DEVICE))
    return model.to(config.DEVICE).eval()


@torch.no_grad()
def run_inference(model, loader, tta=False):
    """Predict on loader; with TTA averages logits over 4 flipped views."""
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(config.DEVICE)
        if tta:
            logits  = model(imgs)
            logits += model(torch.flip(imgs, [3]))      # horizontal flip
            logits += model(torch.flip(imgs, [2]))      # vertical flip
            logits += model(torch.flip(imgs, [2, 3]))   # both
            preds   = logits.argmax(1).cpu().numpy()
        else:
            preds = model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def eval_and_record(model, loader, name, tta, tta_suffix):
    """Run inference, print report, save confusion matrix, return metrics dict."""
    preds, labels = run_inference(model, loader, tta=tta)
    print(classification_report(labels, preds,
                                target_names=config.TARGET_CLASSES, digits=4))
    print(f"Overall accuracy: {accuracy_score(labels, preds):.4f}")
    plot_confusion_matrix(
        preds, labels, config.TARGET_CLASSES,
        title=f"{name}{' (TTA)' if tta else ''} — Test Set",
        save_path=OUTPUT_DIR / f"confusion_matrix_{name}{tta_suffix}.png",
    )
    return metrics_dict(preds, labels, config.TARGET_CLASSES)


def main():
    parser = argparse.ArgumentParser(description="Test set evaluation")
    parser.add_argument("--tta", action="store_true",
                        help="Test-time augmentation: average 4 flipped views")
    args = parser.parse_args()

    tta_suffix = "_tta" if args.tta else ""
    if args.tta:
        print("TTA enabled: orig + hflip + vflip + both")

    loader = DataLoader(
        TestSetDataset(transform=val_transforms),
        batch_size=64, shuffle=False, num_workers=2, pin_memory=True,
    )

    summary = {}

    print("\n── Approach A (EfficientNet-B0, weighted CE) ──")
    summary[f"approach_a{tta_suffix}"] = eval_and_record(
        load_model_a(), loader, "approach_a", args.tta, tta_suffix)

    b_variants = [
        "resnet18",
        "resnet50",
        "efficientnet_b0",
        "efficientnet_b0_unfrozen",
        "efficientnet_b0_unfrozen_weighted",
        "efficientnet_b0_unfrozen_all",
    ]
    for variant in b_variants:
        ckpt = config.ROOT_DIR / f"checkpoints_b_{variant}" / "best_model_b.pth"
        if not ckpt.exists():
            print(f"\n── Approach B ({variant}) — checkpoint not found, skipping ──")
            continue
        print(f"\n── Approach B SupCon ({variant}) ──")
        summary[f"approach_b_{variant}{tta_suffix}"] = eval_and_record(
            load_model_b(variant), loader, f"approach_b_{variant}",
            args.tta, tta_suffix)

    out = OUTPUT_DIR / f"results_summary{tta_suffix}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll results saved to {out}")


if __name__ == "__main__":
    main()
