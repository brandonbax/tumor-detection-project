import os
import random
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

import torchmetrics
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)
from monai.losses import DiceCELoss

import config


def set_seed(seed: int = config.SEED):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    # GPU will be used in the coursework, but fallback to CPU if not available
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SegmentationMetrics:
    """
    Wrapper around torchmetrics for segmentation evaluation.

    Accumulates predictions over batches on-device, then computes
    per-class and mean Dice, IoU, pixel accuracy, precision, and recall.
    """

    def __init__(self,
                 num_classes: int = config.NUM_CLASSES,
                 device: torch.device = None):
        self.num_classes = num_classes
        self.device = device or torch.device("cpu")

        self._metrics = torchmetrics.MetricCollection({
            "dice": MulticlassF1Score(
                num_classes=num_classes, average=None),
            "iou": MulticlassJaccardIndex(
                num_classes=num_classes, average=None),
            "pixel_accuracy": MulticlassAccuracy(
                num_classes=num_classes, average="micro"),
            "precision": MulticlassPrecision(
                num_classes=num_classes, average=None),
            "recall": MulticlassRecall(
                num_classes=num_classes, average=None),
            "confusion": MulticlassConfusionMatrix(
                num_classes=num_classes),
        }).to(self.device)

    def reset(self):
        self._metrics.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Parameters
        ----------
        preds   : (B, C, H, W) logits  or  (B, H, W) class indices
        targets : (B, H, W) class indices
        """
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)

        # torchmetrics expects flat tensors for classification metrics
        self._metrics.update(preds.flatten(), targets.flatten())

    def compute(self) -> Dict:
        """Compute all metrics and return raw torchmetrics output."""
        return self._metrics.compute()

    def mean_dice(self) -> float:
        computed = self.compute()
        return float(computed["dice"].mean())

    def mean_iou(self) -> float:
        computed = self.compute()
        return float(computed["iou"].mean())

    @property
    def confusion(self) -> np.ndarray:
        """
        Return the confusion matrix as a numpy array (for visualisation).
        """
        computed = self.compute()
        return computed["confusion"].cpu().numpy().astype(np.int64)

    def summary(self, class_names: List[str] = None) -> Dict:
        """
        Return a flat dict of all metrics, keyed for easy logging.
        """
        if class_names is None:
            class_names = config.CLASS_NAMES
        computed = self.compute()

        dice = computed["dice"].cpu().numpy()
        iou = computed["iou"].cpu().numpy()
        prec = computed["precision"].cpu().numpy()
        rec = computed["recall"].cpu().numpy()

        result = {
            "pixel_accuracy": float(computed["pixel_accuracy"]),
            "mean_dice": float(dice.mean()),
            "mean_iou": float(iou.mean()),
        }
        for i, name in enumerate(class_names):
            result[f"dice_{name}"] = float(dice[i])
            result[f"iou_{name}"] = float(iou[i])
            result[f"precision_{name}"] = float(prec[i])
            result[f"recall_{name}"] = float(rec[i])

        return result

    def print_summary(self, class_names: List[str] = None):
        s = self.summary(class_names)
        names = class_names or config.CLASS_NAMES
        print("=" * 55)
        print(f"  Pixel Accuracy : {s['pixel_accuracy']:.4f}")
        print(f"  Mean Dice      : {s['mean_dice']:.4f}")
        print(f"  Mean IoU       : {s['mean_iou']:.4f}")
        print("-" * 55)
        print(f"  {'Class':<12} {'Dice':>8} {'IoU':>8} "
              f"{'Prec':>8} {'Recall':>8}")
        for name in names:
            print(f"  {name:<12} "
                  f"{s[f'dice_{name}']:>8.4f} "
                  f"{s[f'iou_{name}']:>8.4f} "
                  f"{s[f'precision_{name}']:>8.4f} "
                  f"{s[f'recall_{name}']:>8.4f}")
        print("=" * 55)


class SegmentationCriterion(nn.Module):
    """
    Thin wrapper around MONAI's DiceCELoss that accepts targets
    as (B, H, W) integer labels (the standard PyTorch convention).

    MONAI expects targets as (B, 1, H, W), so we unsqueeze here.
    """

    def __init__(self,
                 num_classes: int = config.NUM_CLASSES,
                 weight: Optional[torch.Tensor] = None,
                 lambda_dice: float = 0.5,
                 lambda_ce: float = 0.5):
        super().__init__()
        self.loss = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            weight=weight,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits  : (B, C, H, W)  raw model output
        targets : (B, H, W)     integer class labels
        """
        targets = targets.unsqueeze(1)      # (B, 1, H, W)
        return self.loss(logits, targets)


def get_criterion(weight: Optional[torch.Tensor] = None,
                  lambda_dice: float = 0.5,
                  lambda_ce: float = 0.5) -> SegmentationCriterion:
    """
    Factory that returns a MONAI-backed Dice + CE criterion.
    """
    return SegmentationCriterion(
        weight=weight,
        lambda_dice=lambda_dice,
        lambda_ce=lambda_ce,
    )


def compute_class_weights(dataset,
                          num_classes: int = config.NUM_CLASSES,
                          max_samples: int = 100) -> torch.Tensor:
    """
    Estimate inverse-frequency class weights from a subset of the dataset.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    n = min(len(dataset), max_samples)
    for i in range(n):
        _, mask = dataset[i]
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        for c in range(num_classes):
            counts[c] += (mask == c).sum()
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


PALETTE = np.array([
    [0,   0,   255],   # 0 = Other
    [255, 0,   0],     # 1 = Tumor
    [0,   255, 0],     # 2 = Stroma
], dtype=np.uint8)

def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised image tensor back to uint8 RGB for display."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array(config.DATASET_MEAN)
    std = np.array(config.DATASET_STD)
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert a (H, W) integer mask to (H, W, 3) RGB using PALETTE."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(PALETTE)):
        rgb[mask == c] = PALETTE[c]
    return rgb


def save_prediction_grid(images: torch.Tensor,
                         masks_true: torch.Tensor,
                         masks_pred: torch.Tensor,
                         save_path: str,
                         num_samples: int = 4):
    """Save a grid of, image, ground-truth, and prediction side by side."""
    n = min(num_samples, images.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        img = denormalize(images[i])
        gt = mask_to_rgb(masks_true[i].cpu().numpy())
        pred = mask_to_rgb(masks_pred[i].cpu().numpy())

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved prediction grid → {save_path}")


def save_confusion_matrix(metrics: SegmentationMetrics,
                          save_path: str,
                          class_names: List[str] = None):
    """Save a normalised confusion matrix heatmap."""
    if class_names is None:
        class_names = config.CLASS_NAMES

    cm = metrics.confusion.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums > 0,
                        out=np.zeros_like(cm))

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalised Confusion Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix → {save_path}")


def save_training_curves(train_losses: List[float],
                         val_losses: List[float],
                         val_dices: List[float],
                         save_path: str):
    """Plot and save training & validation loss / Dice curves."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_dices, label="Val Mean Dice", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice")
    ax2.set_title("Validation Dice")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves → {save_path}")


def save_reconstruction_grid(originals: torch.Tensor,
                             reconstructions: torch.Tensor,
                             save_path: str,
                             num_samples: int = 4):
    """Save original vs. reconstruction pairs (for autoencoder)."""
    n = min(num_samples, originals.size(0))
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        orig = denormalize(originals[i])
        recon = reconstructions[i].cpu().numpy().transpose(1, 2, 0)
        recon = np.clip(recon * 255, 0, 255).astype(np.uint8)

        axes[i, 0].imshow(orig)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(recon)
        axes[i, 1].set_title("Reconstruction")
        axes[i, 1].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reconstruction grid → {save_path}")