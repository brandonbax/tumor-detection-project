"""
utils.py
--------
Shared helpers for Task 2: plotting, metrics, and feature extraction.
"""
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score,
)
from sklearn.manifold import TSNE
import config


def save_training_curves(history: dict, save_path: Path, title_prefix: str = ""):
    """Plot loss and accuracy curves from a training history dict and save to disk."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.set_title(f"{title_prefix} — Loss")
    ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"],   label="Val")
    ax2.axhline(0.7083, color="r", linestyle="--", label="Baseline (0.7083)")
    ax2.set_title(f"{title_prefix} — Accuracy")
    ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(preds, labels, class_names, title, save_path):
    """Save a labelled confusion-matrix heatmap."""
    cm     = confusion_matrix(labels, preds)
    ticks  = [c.replace("nuclei_", "") for c in class_names]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=ticks, yticklabels=ticks)
    plt.title(title); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


@torch.no_grad()
def extract_features(encoder, loader):
    """Run encoder over the full loader; return (features, labels) as numpy arrays."""
    encoder.eval()
    feats, labels = [], []
    for imgs, lbls in loader:
        f = encoder(imgs.to(config.DEVICE)).flatten(1).cpu().numpy()
        feats.append(f)
        labels.extend(lbls.numpy())
    return np.concatenate(feats), np.array(labels)


def plot_tsne(features, labels, class_names, title, save_path):
    """Fit t-SNE on features and save a scatter plot coloured by class."""
    print("Running t-SNE...")
    reduced = TSNE(n_components=2, perplexity=40, random_state=42,
                   max_iter=1000).fit_transform(features)
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        mask = labels == i
        plt.scatter(reduced[mask, 0], reduced[mask, 1],
                    label=name, alpha=0.5, s=8)
    plt.legend(markerscale=2); plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  t-SNE saved to {save_path}")


def metrics_dict(preds, labels, class_names):
    """Return accuracy + per-class precision/recall/F1 as a nested dict."""
    return {
        "accuracy":        round(float(accuracy_score(labels, preds)), 4),
        "macro_precision": round(float(precision_score(labels, preds, average="macro", zero_division=0)), 4),
        "macro_recall":    round(float(recall_score(labels, preds, average="macro", zero_division=0)), 4),
        "macro_f1":        round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
        "per_class": {
            name: {
                "precision": round(float(precision_score(labels, preds, labels=[i], average="macro", zero_division=0)), 4),
                "recall":    round(float(recall_score(labels, preds, labels=[i], average="macro", zero_division=0)), 4),
                "f1":        round(float(f1_score(labels, preds, labels=[i], average="macro", zero_division=0)), 4),
            }
            for i, name in enumerate(class_names)
        },
    }
