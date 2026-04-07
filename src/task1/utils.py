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
from tqdm import tqdm

import torchmetrics
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
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
    Segmentation evaluation with per-image Dice and IoU averaging.

    Dice and IoU are computed per-image, per-class, then averaged across
    images (skipping images where a class is absent from the ground
    truth).  This avoids large-region dominance that occurs with global
    TP/FP/FN accumulation.

    Pixel accuracy, precision, recall, and the confusion matrix are
    still accumulated globally via torchmetrics.
    """

    def __init__(self,
                 num_classes: int = config.NUM_CLASSES,
                 device: torch.device = None):
        self.num_classes = num_classes
        self.device = device or torch.device("cpu")

        # Global metrics (torchmetrics)
        self._global_metrics = torchmetrics.MetricCollection({
            "pixel_accuracy": MulticlassAccuracy(
                num_classes=num_classes, average="micro"),
            "precision": MulticlassPrecision(
                num_classes=num_classes, average=None),
            "recall": MulticlassRecall(
                num_classes=num_classes, average=None),
            "confusion": MulticlassConfusionMatrix(
                num_classes=num_classes),
        }).to(self.device)

        # Per-image Dice / IoU accumulators
        self._dice_sums = torch.zeros(num_classes, device=self.device)
        self._iou_sums = torch.zeros(num_classes, device=self.device)
        self._class_image_counts = torch.zeros(num_classes,
                                               device=self.device)

    def reset(self):
        self._global_metrics.reset()
        self._dice_sums.zero_()
        self._iou_sums.zero_()
        self._class_image_counts.zero_()

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Parameters
        ----------
        preds   : (B, C, H, W) logits  or  (B, H, W) class indices
        targets : (B, H, W) class indices
        """
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)

        # Global metrics
        self._global_metrics.update(preds.flatten(), targets.flatten())

        # Per-image Dice and IoU
        for i in range(preds.size(0)):
            pred_i = preds[i]
            target_i = targets[i]

            for c in range(self.num_classes):
                gt_c = (target_i == c)
                if not gt_c.any():
                    continue

                pred_c = (pred_i == c)
                tp = (pred_c & gt_c).sum().float()
                fp = (pred_c & ~gt_c).sum().float()
                fn = (~pred_c & gt_c).sum().float()

                dice_denom = 2.0 * tp + fp + fn
                if dice_denom > 0:
                    self._dice_sums[c] += 2.0 * tp / dice_denom

                iou_denom = tp + fp + fn
                if iou_denom > 0:
                    self._iou_sums[c] += tp / iou_denom

                self._class_image_counts[c] += 1

    def _per_class_dice(self) -> torch.Tensor:
        counts = self._class_image_counts.clamp(min=1)
        return self._dice_sums / counts

    def _per_class_iou(self) -> torch.Tensor:
        counts = self._class_image_counts.clamp(min=1)
        return self._iou_sums / counts

    def mean_dice(self) -> float:
        return float(self._per_class_dice().mean())

    def mean_iou(self) -> float:
        return float(self._per_class_iou().mean())

    @property
    def confusion(self) -> np.ndarray:
        """
        Return the confusion matrix as a numpy array (for visualisation).
        """
        computed = self._global_metrics.compute()
        return computed["confusion"].cpu().numpy().astype(np.int64)

    def summary(self, class_names: List[str] = None) -> Dict:
        """
        Return a flat dict of all metrics, keyed for easy logging.
        """
        if class_names is None:
            class_names = config.CLASS_NAMES
        computed = self._global_metrics.compute()

        dice = self._per_class_dice().cpu().numpy()
        iou = self._per_class_iou().cpu().numpy()
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
                 lambda_dice: float = config.LAMBDA_DICE,
                 lambda_ce: float = config.LAMBDA_CE):
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


class DeepSupervisionCriterion(nn.Module):
    """Wraps a base criterion to support deep supervision.

    During training the model returns ``(main_logits, [side_logits, ...])``.
    The total loss is::

        loss = base_criterion(main) + side_weight * mean(base_criterion(side_i))

    During evaluation the model returns only ``main_logits`` and this
    criterion behaves identically to the base criterion.
    """

    def __init__(self, base_criterion: nn.Module, side_weight: float = 0.5):
        super().__init__()
        self.base = base_criterion
        self.side_weight = side_weight

    def forward(self, model_output, targets: torch.Tensor):
        if isinstance(model_output, tuple):
            main_logits, side_logits = model_output
            loss = self.base(main_logits, targets)
            if side_logits:
                side_loss = sum(self.base(s, targets) for s in side_logits)
                loss = loss + self.side_weight * side_loss / len(side_logits)
            return loss
        return self.base(model_output, targets)


def get_criterion(weight: Optional[torch.Tensor] = None,
                  lambda_dice: float = config.LAMBDA_DICE,
                  lambda_ce: float = config.LAMBDA_CE,
                  deep_supervision: bool = True,
                  side_weight: float = 0.5) -> nn.Module:
    """
    Factory that returns a MONAI-backed Dice + CE criterion,
    optionally wrapped for deep supervision.
    """
    base = SegmentationCriterion(
        weight=weight,
        lambda_dice=lambda_dice,
        lambda_ce=lambda_ce,
    )
    if deep_supervision:
        return DeepSupervisionCriterion(base, side_weight=side_weight)
    return base


def compute_class_weights(dataset,
                          num_classes: int = config.NUM_CLASSES,
                          max_samples: int = 100,
                          dampen: bool = False) -> torch.Tensor:
    """
    Estimate inverse-frequency class weights from a subset of the dataset.

    Parameters
    ----------
    dampen : bool
        If True, use 1/sqrt(count) instead of 1/count for softer
        rebalancing that helps rare classes without crippling
        dominant ones.
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
    if dampen:
        weights = 1.0 / np.sqrt(counts)
    else:
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


def save_class_distribution(dataset,
                            save_path: str,
                            num_classes: int = config.NUM_CLASSES,
                            class_names: List[str] = None):
    """
    Compute and plot the pixel-level class distribution of a dataset.

    Produces a bar chart showing the percentage of pixels per class,
    useful for illustrating class imbalance.
    """
    if class_names is None:
        class_names = config.CLASS_NAMES

    counts = np.zeros(num_classes, dtype=np.float64)
    for i in tqdm(range(len(dataset)), desc="  Class distribution"):
        _, mask = dataset[i]
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        for c in range(num_classes):
            counts[c] += (mask == c).sum()

    total = counts.sum()
    percentages = counts / total * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(class_names, percentages, color=[
        f"#{PALETTE[i][0]:02x}{PALETTE[i][1]:02x}{PALETTE[i][2]:02x}"
        for i in range(num_classes)
    ])
    for bar, pct, cnt in zip(bars, percentages, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%\n({int(cnt):,} px)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Percentage of Pixels")
    ax.set_title("Class Distribution (Training Set)")
    ax.set_ylim(0, max(percentages) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved class distribution → {save_path}")

    return {name: float(pct) for name, pct in zip(class_names, percentages)}


def save_model_comparison_grid(images: torch.Tensor,
                               masks_true: torch.Tensor,
                               model_preds: Dict[str, torch.Tensor],
                               save_path: str,
                               num_samples: int = 4):
    """
    Save a side-by-side grid: Image | Ground Truth | Model1 | Model2 | ...

    Parameters
    ----------
    images      : (B, 3, H, W) normalised image tensors
    masks_true  : (B, H, W) ground-truth masks
    model_preds : dict mapping model name to (B, H, W) predicted masks
    """
    n = min(num_samples, images.size(0))
    model_names = list(model_preds.keys())
    ncols = 2 + len(model_names)  # image + gt + each model

    fig, axes = plt.subplots(n, ncols, figsize=(4 * ncols, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        img = denormalize(images[i])
        gt = mask_to_rgb(masks_true[i].cpu().numpy())

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        for j, name in enumerate(model_names):
            pred = mask_to_rgb(model_preds[name][i].cpu().numpy())
            axes[i, 2 + j].imshow(pred)
            axes[i, 2 + j].set_title(name)
            axes[i, 2 + j].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved model comparison grid → {save_path}")


def save_metrics_comparison_chart(all_results: Dict[str, Dict],
                                  save_path: str,
                                  class_names: List[str] = None):
    """
    Save a grouped bar chart comparing Dice and IoU per class across models.
    """
    if class_names is None:
        class_names = config.CLASS_NAMES

    model_names = list(all_results.keys())
    n_models = len(model_names)
    n_classes = len(class_names)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(n_classes)
    width = 0.8 / n_models

    for idx, name in enumerate(model_names):
        res = all_results[name]
        dice_vals = [res.get(f"dice_{cn}", 0) for cn in class_names]
        iou_vals = [res.get(f"iou_{cn}", 0) for cn in class_names]

        offset = (idx - (n_models - 1) / 2) * width
        ax1.bar(x + offset, dice_vals, width, label=name)
        ax2.bar(x + offset, iou_vals, width, label=name)

    for ax, title in [(ax1, "Dice per Class"), (ax2, "IoU per Class")]:
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_ylim(0, 1.0)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved metrics comparison chart → {save_path}")