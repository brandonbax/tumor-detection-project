"""
test_train_autoencoder.py
-------------------------
Unit tests for src/task1/train_autoencoder.py

Covers:
  - parse_args         (CLI argument parsing and defaults)
  - train_one_epoch    (single training step on synthetic data)
  - validate           (evaluation loop, returns average MSE)
  - main               (end-to-end training with mocked data loaders)
"""

import os
import argparse

import torch
import torch.nn as nn
import pytest
from unittest.mock import patch
from torch.utils.data import DataLoader, TensorDataset

import config
from task1.train_autoencoder import parse_args, train_one_epoch, validate


# ── Constants ────────────────────────────────────────

PATCH = 16          # small spatial size for fast tests
BATCH = 2
IN_CH = 3


# ── Fixtures ─────────────────────────────────────────

@pytest.fixture
def tiny_autoencoder():
    """A small Autoencoder that runs quickly on CPU."""
    from task1.models import Autoencoder
    return Autoencoder(in_channels=IN_CH, features=[8, 16, 32])


@pytest.fixture
def criterion():
    return nn.MSELoss()


@pytest.fixture
def synthetic_loader():
    """DataLoader yielding images only (no masks) — matches get_unlabelled_loaders."""
    images = torch.randn(4, IN_CH, PATCH, PATCH)
    ds = TensorDataset(images)
    # Wrap so iteration yields plain tensors, not tuples
    class UnlabelledLoader:
        def __init__(self, loader):
            self._loader = loader
        def __iter__(self):
            for (batch,) in self._loader:
                yield batch
        def __len__(self):
            return len(self._loader)
        @property
        def dataset(self):
            return self._loader.dataset
    return UnlabelledLoader(DataLoader(ds, batch_size=BATCH, shuffle=False))


@pytest.fixture
def device():
    return torch.device("cpu")


# ═══════════════════════════════════════════════
#  parse_args
# ═══════════════════════════════════════════════

class TestParseArgs:

    def test_defaults(self):
        """All defaults should match config values."""
        with patch("sys.argv", ["train_autoencoder.py"]):
            args = parse_args()
        assert args.epochs == config.AE_EPOCHS
        assert args.batch_size == config.AE_BATCH_SIZE
        assert args.lr == config.AE_LR
        assert args.weight_decay == config.AE_WEIGHT_DECAY
        assert args.patch_size == config.PATCH_SIZE
        assert args.features == config.AE_FEATURES
        assert args.data_root == config.DATASET_ROOT
        assert args.checkpoint_dir == config.CHECKPOINT_DIR
        assert args.results_dir == config.RESULTS_DIR
        assert args.seed == config.SEED

    def test_custom_epochs(self):
        with patch("sys.argv", ["train_autoencoder.py", "--epochs", "10"]):
            args = parse_args()
        assert args.epochs == 10

    def test_custom_lr(self):
        with patch("sys.argv", ["train_autoencoder.py", "--lr", "0.001"]):
            args = parse_args()
        assert args.lr == pytest.approx(0.001)

    def test_custom_features(self):
        with patch("sys.argv",
                   ["train_autoencoder.py", "--features", "32", "64", "128"]):
            args = parse_args()
        assert args.features == [32, 64, 128]

    def test_custom_batch_size(self):
        with patch("sys.argv",
                   ["train_autoencoder.py", "--batch_size", "4"]):
            args = parse_args()
        assert args.batch_size == 4

    def test_custom_data_root(self):
        with patch("sys.argv",
                   ["train_autoencoder.py", "--data_root", "/tmp/mydata"]):
            args = parse_args()
        assert args.data_root == "/tmp/mydata"

    def test_custom_seed(self):
        with patch("sys.argv", ["train_autoencoder.py", "--seed", "123"]):
            args = parse_args()
        assert args.seed == 123

    def test_custom_weight_decay(self):
        with patch("sys.argv",
                   ["train_autoencoder.py", "--weight_decay", "0.01"]):
            args = parse_args()
        assert args.weight_decay == pytest.approx(0.01)


# ═══════════════════════════════════════════════
#  train_one_epoch
# ═══════════════════════════════════════════════

class TestTrainOneEpoch:

    def test_returns_float_loss(self, tiny_autoencoder, criterion,
                                synthetic_loader, device):
        optimizer = torch.optim.SGD(tiny_autoencoder.parameters(), lr=0.01)
        loss = train_one_epoch(tiny_autoencoder, synthetic_loader, criterion,
                               optimizer, device)
        assert isinstance(loss, float)
        assert loss > 0

    def test_model_in_train_mode(self, tiny_autoencoder, criterion,
                                  synthetic_loader, device):
        """After train_one_epoch, model should still be in train mode."""
        optimizer = torch.optim.SGD(tiny_autoencoder.parameters(), lr=0.01)
        train_one_epoch(tiny_autoencoder, synthetic_loader, criterion,
                        optimizer, device)
        assert tiny_autoencoder.training is True

    def test_weights_update(self, tiny_autoencoder, criterion,
                            synthetic_loader, device):
        """Model parameters should change after one training step."""
        optimizer = torch.optim.SGD(tiny_autoencoder.parameters(), lr=0.01)
        params_before = {
            n: p.clone() for n, p in tiny_autoencoder.named_parameters()
        }
        train_one_epoch(tiny_autoencoder, synthetic_loader, criterion,
                        optimizer, device)
        any_changed = any(
            not torch.equal(params_before[n], p)
            for n, p in tiny_autoencoder.named_parameters()
        )
        assert any_changed

    def test_loss_decreases_over_epochs(self, tiny_autoencoder, criterion,
                                        synthetic_loader, device):
        """Loss should generally decrease over multiple epochs on fixed data."""
        optimizer = torch.optim.SGD(tiny_autoencoder.parameters(), lr=0.01)
        losses = []
        for _ in range(5):
            loss = train_one_epoch(tiny_autoencoder, synthetic_loader,
                                   criterion, optimizer, device)
            losses.append(loss)
        assert losses[-1] < losses[0]

    def test_reconstruction_output_range(self, tiny_autoencoder,
                                          synthetic_loader, device):
        """The autoencoder decoder uses sigmoid, so output should be in [0, 1]."""
        tiny_autoencoder.eval()
        with torch.no_grad():
            for images in synthetic_loader:
                images = images.to(device)
                recon = tiny_autoencoder(images)
                assert recon.min() >= 0.0
                assert recon.max() <= 1.0
                break


# ═══════════════════════════════════════════════
#  validate
# ═══════════════════════════════════════════════

class TestValidate:

    def test_returns_float_loss(self, tiny_autoencoder, criterion,
                                synthetic_loader, device):
        val_loss = validate(tiny_autoencoder, synthetic_loader,
                            criterion, device)
        assert isinstance(val_loss, float)
        assert val_loss > 0

    def test_model_in_eval_mode(self, tiny_autoencoder, criterion,
                                 synthetic_loader, device):
        """validate() should set the model to eval mode."""
        validate(tiny_autoencoder, synthetic_loader, criterion, device)
        assert tiny_autoencoder.training is False

    def test_no_grad_context(self, tiny_autoencoder, criterion,
                              synthetic_loader, device):
        """Parameters should not accumulate gradients during validation."""
        validate(tiny_autoencoder, synthetic_loader, criterion, device)
        for param in tiny_autoencoder.parameters():
            assert param.grad is None or torch.all(param.grad == 0)

    def test_val_loss_consistent(self, tiny_autoencoder, criterion,
                                  synthetic_loader, device):
        """Calling validate twice on same data should give the same loss."""
        loss1 = validate(tiny_autoencoder, synthetic_loader, criterion, device)
        loss2 = validate(tiny_autoencoder, synthetic_loader, criterion, device)
        assert loss1 == pytest.approx(loss2)


# ═══════════════════════════════════════════════
#  main (integration, end-to-end with mocks)
# ═══════════════════════════════════════════════

class TestMain:

    def _make_loader(self):
        """Create a synthetic unlabelled loader for main() tests."""
        images = torch.randn(4, IN_CH, PATCH, PATCH)
        ds = TensorDataset(images)

        class UnlabelledLoader:
            def __init__(self, loader):
                self._loader = loader
            def __iter__(self):
                for (batch,) in self._loader:
                    yield batch
            def __len__(self):
                return len(self._loader)
            @property
            def dataset(self):
                return self._loader.dataset

        return UnlabelledLoader(DataLoader(ds, batch_size=BATCH, shuffle=False))

    def test_end_to_end_training(self, tmp_path):
        """Run main() for 2 epochs on synthetic data, verify outputs."""
        from task1.train_autoencoder import main

        ckpt_dir = str(tmp_path / "checkpoints")
        results_dir = str(tmp_path / "results")
        loader = self._make_loader()

        fake_args = argparse.Namespace(
            epochs=2,
            batch_size=BATCH,
            lr=1e-3,
            weight_decay=1e-4,
            patch_size=PATCH,
            features=[8, 16, 32],
            data_root="fake",
            checkpoint_dir=ckpt_dir,
            results_dir=results_dir,
            seed=42,
        )

        with patch("task1.train_autoencoder.parse_args",
                   return_value=fake_args), \
             patch("task1.train_autoencoder.get_unlabelled_loaders",
                   return_value=(loader, loader)), \
             patch("task1.train_autoencoder.get_device",
                   return_value=torch.device("cpu")):
            main()

        # Checkpoint should be saved
        assert os.path.isfile(os.path.join(ckpt_dir, "autoencoder_best.pth"))

        # Training curves should be saved
        assert os.path.isfile(
            os.path.join(results_dir, "ae_training_curves.png"))

        # Reconstruction grid should be saved
        assert os.path.isfile(
            os.path.join(results_dir, "ae_reconstructions.png"))

    def test_checkpoint_contents(self, tmp_path):
        """Verify the saved checkpoint contains the expected keys."""
        from task1.train_autoencoder import main

        ckpt_dir = str(tmp_path / "checkpoints")
        results_dir = str(tmp_path / "results")
        loader = self._make_loader()

        fake_args = argparse.Namespace(
            epochs=1,
            batch_size=BATCH,
            lr=1e-3,
            weight_decay=1e-4,
            patch_size=PATCH,
            features=[8, 16, 32],
            data_root="fake",
            checkpoint_dir=ckpt_dir,
            results_dir=results_dir,
            seed=42,
        )

        with patch("task1.train_autoencoder.parse_args",
                   return_value=fake_args), \
             patch("task1.train_autoencoder.get_unlabelled_loaders",
                   return_value=(loader, loader)), \
             patch("task1.train_autoencoder.get_device",
                   return_value=torch.device("cpu")):
            main()

        ckpt = torch.load(os.path.join(ckpt_dir, "autoencoder_best.pth"),
                          map_location="cpu", weights_only=False)
        assert "epoch" in ckpt
        assert "model_state_dict" in ckpt
        assert "encoder_state_dict" in ckpt
        assert "best_loss" in ckpt
        assert "features" in ckpt
        assert "args" in ckpt
        assert isinstance(ckpt["best_loss"], float)
        assert ckpt["epoch"] >= 1

    def test_best_loss_is_finite(self, tmp_path):
        """After training, best_loss in the checkpoint should be finite."""
        from task1.train_autoencoder import main

        ckpt_dir = str(tmp_path / "checkpoints")
        results_dir = str(tmp_path / "results")
        loader = self._make_loader()

        fake_args = argparse.Namespace(
            epochs=2,
            batch_size=BATCH,
            lr=1e-3,
            weight_decay=1e-4,
            patch_size=PATCH,
            features=[8, 16, 32],
            data_root="fake",
            checkpoint_dir=ckpt_dir,
            results_dir=results_dir,
            seed=42,
        )

        with patch("task1.train_autoencoder.parse_args",
                   return_value=fake_args), \
             patch("task1.train_autoencoder.get_unlabelled_loaders",
                   return_value=(loader, loader)), \
             patch("task1.train_autoencoder.get_device",
                   return_value=torch.device("cpu")):
            main()

        ckpt = torch.load(os.path.join(ckpt_dir, "autoencoder_best.pth"),
                          map_location="cpu", weights_only=False)
        assert ckpt["best_loss"] > 0.0
        assert ckpt["best_loss"] < float("inf")
