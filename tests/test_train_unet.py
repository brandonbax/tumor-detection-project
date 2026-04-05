"""
test_train_unet.py
------------------
Unit tests for src/task1/train_unet.py

Covers:
  - parse_args         (CLI argument parsing and defaults)
  - train_one_epoch    (single training step on synthetic data)
  - validate           (evaluation loop, loss + metrics)
  - main               (end-to-end training with mocked data loaders)
"""

import os
import argparse

import torch
import torch.nn as nn
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader, TensorDataset

import config
from task1.train_unet import parse_args, train_one_epoch, validate


# ── Constants ────────────────────────────────────────

NUM_CLASSES = 3
PATCH = 16          # small spatial size for fast tests
BATCH = 2
IN_CH = 3


# ── Fixtures ─────────────────────────────────────────

@pytest.fixture
def tiny_model():
    """A small UNet that runs quickly on CPU."""
    from task1.models import UNet
    return UNet(in_channels=IN_CH, num_classes=NUM_CLASSES,
                features=[8, 16, 32])


@pytest.fixture
def criterion():
    from task1.utils import get_criterion
    return get_criterion()


@pytest.fixture
def synthetic_loader():
    """DataLoader yielding (images, masks) with deterministic shapes."""
    images = torch.randn(4, IN_CH, PATCH, PATCH)
    masks = torch.randint(0, NUM_CLASSES, (4, PATCH, PATCH))
    ds = TensorDataset(images, masks)
    return DataLoader(ds, batch_size=BATCH, shuffle=False)


@pytest.fixture
def device():
    return torch.device("cpu")


# ═══════════════════════════════════════════════
#  parse_args
# ═══════════════════════════════════════════════

class TestParseArgs:

    def test_defaults(self):
        """All defaults should match config values."""
        with patch("sys.argv", ["train_unet.py"]):
            args = parse_args()
        assert args.epochs == config.UNET_EPOCHS
        assert args.batch_size == config.UNET_BATCH_SIZE
        assert args.lr == config.UNET_LR
        assert args.weight_decay == config.UNET_WEIGHT_DECAY
        assert args.patch_size == config.PATCH_SIZE
        assert args.features == [64, 128, 256, 512, 1024]
        assert args.data_root == config.DATASET_ROOT
        assert args.use_class_weights is False
        assert args.checkpoint_dir == config.CHECKPOINT_DIR
        assert args.results_dir == config.RESULTS_DIR
        assert args.seed == config.SEED

    def test_custom_epochs(self):
        with patch("sys.argv", ["train_unet.py", "--epochs", "10"]):
            args = parse_args()
        assert args.epochs == 10

    def test_custom_lr(self):
        with patch("sys.argv", ["train_unet.py", "--lr", "0.001"]):
            args = parse_args()
        assert args.lr == pytest.approx(0.001)

    def test_custom_features(self):
        with patch("sys.argv",
                   ["train_unet.py", "--features", "32", "64", "128"]):
            args = parse_args()
        assert args.features == [32, 64, 128]

    def test_use_class_weights_flag(self):
        with patch("sys.argv",
                   ["train_unet.py", "--use_class_weights"]):
            args = parse_args()
        assert args.use_class_weights is True

    def test_custom_data_root(self):
        with patch("sys.argv",
                   ["train_unet.py", "--data_root", "/tmp/mydata"]):
            args = parse_args()
        assert args.data_root == "/tmp/mydata"

    def test_custom_seed(self):
        with patch("sys.argv", ["train_unet.py", "--seed", "123"]):
            args = parse_args()
        assert args.seed == 123


# ═══════════════════════════════════════════════
#  train_one_epoch
# ═══════════════════════════════════════════════

class TestTrainOneEpoch:

    def test_returns_float_loss(self, tiny_model, criterion,
                                synthetic_loader, device):
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.01)
        loss = train_one_epoch(tiny_model, synthetic_loader, criterion,
                               optimizer, device)
        assert isinstance(loss, float)
        assert loss > 0

    def test_model_in_train_mode(self, tiny_model, criterion,
                                  synthetic_loader, device):
        """After train_one_epoch, model should still be in train mode."""
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.01)
        train_one_epoch(tiny_model, synthetic_loader, criterion,
                        optimizer, device)
        assert tiny_model.training is True

    def test_weights_update(self, tiny_model, criterion,
                            synthetic_loader, device):
        """Model parameters should change after one training step."""
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.01)
        params_before = {
            n: p.clone() for n, p in tiny_model.named_parameters()
        }
        train_one_epoch(tiny_model, synthetic_loader, criterion,
                        optimizer, device)
        any_changed = any(
            not torch.equal(params_before[n], p)
            for n, p in tiny_model.named_parameters()
        )
        assert any_changed

    def test_loss_decreases_over_epochs(self, tiny_model, criterion,
                                        synthetic_loader, device):
        """Loss should generally decrease over multiple epochs on fixed data."""
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.01)
        losses = []
        for _ in range(5):
            loss = train_one_epoch(tiny_model, synthetic_loader, criterion,
                                   optimizer, device)
            losses.append(loss)
        # The last loss should be smaller than the first
        assert losses[-1] < losses[0]


# ═══════════════════════════════════════════════
#  validate
# ═══════════════════════════════════════════════

class TestValidate:

    def test_returns_loss_and_metrics(self, tiny_model, criterion,
                                      synthetic_loader, device):
        avg_loss, metrics = validate(tiny_model, synthetic_loader,
                                     criterion, device)
        assert isinstance(avg_loss, float)
        assert avg_loss > 0

    def test_model_in_eval_mode(self, tiny_model, criterion,
                                 synthetic_loader, device):
        """validate() should set the model to eval mode."""
        validate(tiny_model, synthetic_loader, criterion, device)
        assert tiny_model.training is False

    def test_metrics_have_dice(self, tiny_model, criterion,
                                synthetic_loader, device):
        _, metrics = validate(tiny_model, synthetic_loader,
                              criterion, device)
        dice = metrics.mean_dice()
        assert isinstance(dice, float)
        assert 0.0 <= dice <= 1.0

    def test_metrics_have_iou(self, tiny_model, criterion,
                               synthetic_loader, device):
        _, metrics = validate(tiny_model, synthetic_loader,
                              criterion, device)
        iou = metrics.mean_iou()
        assert isinstance(iou, float)
        assert 0.0 <= iou <= 1.0

    def test_no_grad_context(self, tiny_model, criterion,
                              synthetic_loader, device):
        """Parameters should not accumulate gradients during validation."""
        validate(tiny_model, synthetic_loader, criterion, device)
        for param in tiny_model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)

    def test_confusion_matrix_shape(self, tiny_model, criterion,
                                     synthetic_loader, device):
        _, metrics = validate(tiny_model, synthetic_loader,
                              criterion, device)
        cm = metrics.confusion
        assert cm.shape == (NUM_CLASSES, NUM_CLASSES)

    def test_summary_keys(self, tiny_model, criterion,
                           synthetic_loader, device):
        _, metrics = validate(tiny_model, synthetic_loader,
                              criterion, device)
        summary = metrics.summary(
            class_names=["Other", "Tumor", "Stroma"]
        )
        assert "pixel_accuracy" in summary
        assert "mean_dice" in summary
        assert "mean_iou" in summary
        assert "dice_Other" in summary
        assert "iou_Tumor" in summary
        assert "precision_Stroma" in summary
        assert "recall_Other" in summary


# ═══════════════════════════════════════════════
#  main (integration, end-to-end with mocks)
# ═══════════════════════════════════════════════

class TestMain:

    def test_end_to_end_training(self, tmp_path, tiny_model, criterion):
        """Run main() for 2 epochs on synthetic data, verify outputs."""
        from task1.train_unet import main

        ckpt_dir = str(tmp_path / "checkpoints")
        results_dir = str(tmp_path / "results")

        # Synthetic data loaders
        images = torch.randn(4, IN_CH, PATCH, PATCH)
        masks = torch.randint(0, NUM_CLASSES, (4, PATCH, PATCH))
        ds = TensorDataset(images, masks)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False)

        fake_args = argparse.Namespace(
            epochs=2,
            batch_size=BATCH,
            lr=1e-3,
            weight_decay=1e-4,
            patch_size=PATCH,
            features=[8, 16, 32],
            data_root="fake",
            use_class_weights=False,
            checkpoint_dir=ckpt_dir,
            results_dir=results_dir,
            seed=42,
        )

        with patch("task1.train_unet.parse_args", return_value=fake_args), \
             patch("task1.train_unet.get_segmentation_loaders",
                   return_value=(loader, loader, loader)), \
             patch("task1.train_unet.get_device",
                   return_value=torch.device("cpu")):
            main()

        # Checkpoint should be saved
        assert os.path.isfile(os.path.join(ckpt_dir, "unet_best.pth"))

        # Training curves should be saved
        assert os.path.isfile(
            os.path.join(results_dir, "unet_training_curves.png"))

        # Prediction grid should be saved
        assert os.path.isfile(
            os.path.join(results_dir, "unet_val_predictions.png"))

    def test_checkpoint_contents(self, tmp_path):
        """Verify the saved checkpoint contains the expected keys."""
        from task1.train_unet import main

        ckpt_dir = str(tmp_path / "checkpoints")
        results_dir = str(tmp_path / "results")

        images = torch.randn(4, IN_CH, PATCH, PATCH)
        masks = torch.randint(0, NUM_CLASSES, (4, PATCH, PATCH))
        ds = TensorDataset(images, masks)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False)

        fake_args = argparse.Namespace(
            epochs=1,
            batch_size=BATCH,
            lr=1e-3,
            weight_decay=1e-4,
            patch_size=PATCH,
            features=[8, 16, 32],
            data_root="fake",
            use_class_weights=False,
            checkpoint_dir=ckpt_dir,
            results_dir=results_dir,
            seed=42,
        )

        with patch("task1.train_unet.parse_args", return_value=fake_args), \
             patch("task1.train_unet.get_segmentation_loaders",
                   return_value=(loader, loader, loader)), \
             patch("task1.train_unet.get_device",
                   return_value=torch.device("cpu")):
            main()

        ckpt = torch.load(os.path.join(ckpt_dir, "unet_best.pth"),
                          map_location="cpu", weights_only=False)
        assert "epoch" in ckpt
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "best_dice" in ckpt
        assert "args" in ckpt
        assert isinstance(ckpt["best_dice"], float)
        assert ckpt["epoch"] >= 1

    def test_best_dice_improves(self, tmp_path):
        """After training, best_dice in the checkpoint should be > 0."""
        from task1.train_unet import main

        ckpt_dir = str(tmp_path / "checkpoints")
        results_dir = str(tmp_path / "results")

        images = torch.randn(4, IN_CH, PATCH, PATCH)
        masks = torch.randint(0, NUM_CLASSES, (4, PATCH, PATCH))
        ds = TensorDataset(images, masks)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False)

        fake_args = argparse.Namespace(
            epochs=2,
            batch_size=BATCH,
            lr=1e-3,
            weight_decay=1e-4,
            patch_size=PATCH,
            features=[8, 16, 32],
            data_root="fake",
            use_class_weights=False,
            checkpoint_dir=ckpt_dir,
            results_dir=results_dir,
            seed=42,
        )

        with patch("task1.train_unet.parse_args", return_value=fake_args), \
             patch("task1.train_unet.get_segmentation_loaders",
                   return_value=(loader, loader, loader)), \
             patch("task1.train_unet.get_device",
                   return_value=torch.device("cpu")):
            main()

        ckpt = torch.load(os.path.join(ckpt_dir, "unet_best.pth"),
                          map_location="cpu", weights_only=False)
        assert ckpt["best_dice"] > 0.0
