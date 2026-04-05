"""
test_train_ae_seg.py
--------------------
Unit tests for src/task1/train_ae_seg.py

Covers:
  - parse_args         (CLI argument parsing and defaults)
  - train_one_epoch    (single training step with frozen encoder)
  - validate           (evaluation loop, loss + metrics)
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
from task1.train_ae_seg import parse_args, train_one_epoch, validate


# ── Constants ────────────────────────────────────────

NUM_CLASSES = 3
PATCH = 16          # small spatial size for fast tests
BATCH = 2
IN_CH = 3
FEATURES = [8, 16, 32]


# ── Fixtures ─────────────────────────────────────────

@pytest.fixture
def tiny_ae_seg_model():
    """A small AESegmentationModel with frozen encoder, fast on CPU."""
    from task1.models import AEEncoder, AESegmentationModel
    encoder = AEEncoder(in_channels=IN_CH, features=FEATURES)
    return AESegmentationModel(encoder, num_classes=NUM_CLASSES,
                               features=FEATURES, freeze_encoder=True)


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
        with patch("sys.argv", ["train_ae_seg.py"]):
            args = parse_args()
        assert args.epochs == config.AE_SEG_EPOCHS
        assert args.batch_size == config.AE_SEG_BATCH_SIZE
        assert args.lr == config.AE_SEG_LR
        assert args.weight_decay == config.AE_SEG_WEIGHT_DECAY
        assert args.patch_size == config.PATCH_SIZE
        assert args.features == [64, 128, 256, 512]
        assert args.data_root == config.DATASET_ROOT
        assert args.use_class_weights is False
        assert args.checkpoint_dir == config.CHECKPOINT_DIR
        assert args.results_dir == config.RESULTS_DIR
        assert args.seed == config.SEED
        assert args.ae_checkpoint == os.path.join(
            config.CHECKPOINT_DIR, "autoencoder_best.pth")

    def test_custom_epochs(self):
        with patch("sys.argv", ["train_ae_seg.py", "--epochs", "10"]):
            args = parse_args()
        assert args.epochs == 10

    def test_custom_lr(self):
        with patch("sys.argv", ["train_ae_seg.py", "--lr", "0.001"]):
            args = parse_args()
        assert args.lr == pytest.approx(0.001)

    def test_custom_features(self):
        with patch("sys.argv",
                   ["train_ae_seg.py", "--features", "32", "64", "128"]):
            args = parse_args()
        assert args.features == [32, 64, 128]

    def test_use_class_weights_flag(self):
        with patch("sys.argv", ["train_ae_seg.py", "--use_class_weights"]):
            args = parse_args()
        assert args.use_class_weights is True

    def test_custom_ae_checkpoint(self):
        with patch("sys.argv",
                   ["train_ae_seg.py", "--ae_checkpoint", "/tmp/ae.pth"]):
            args = parse_args()
        assert args.ae_checkpoint == "/tmp/ae.pth"

    def test_custom_data_root(self):
        with patch("sys.argv",
                   ["train_ae_seg.py", "--data_root", "/tmp/mydata"]):
            args = parse_args()
        assert args.data_root == "/tmp/mydata"

    def test_custom_seed(self):
        with patch("sys.argv", ["train_ae_seg.py", "--seed", "123"]):
            args = parse_args()
        assert args.seed == 123


# ═══════════════════════════════════════════════
#  train_one_epoch
# ═══════════════════════════════════════════════

class TestTrainOneEpoch:

    def test_returns_float_loss(self, tiny_ae_seg_model, criterion,
                                synthetic_loader, device):
        optimizer = torch.optim.SGD(
            tiny_ae_seg_model.seg_decoder.parameters(), lr=0.01)
        loss = train_one_epoch(tiny_ae_seg_model, synthetic_loader, criterion,
                               optimizer, device)
        assert isinstance(loss, float)
        assert loss > 0

    def test_encoder_stays_frozen(self, tiny_ae_seg_model, criterion,
                                   synthetic_loader, device):
        """Encoder parameters must not change during training."""
        encoder_before = {
            n: p.clone()
            for n, p in tiny_ae_seg_model.encoder.named_parameters()
        }
        optimizer = torch.optim.SGD(
            tiny_ae_seg_model.seg_decoder.parameters(), lr=0.01)
        train_one_epoch(tiny_ae_seg_model, synthetic_loader, criterion,
                        optimizer, device)
        for n, p in tiny_ae_seg_model.encoder.named_parameters():
            assert torch.equal(encoder_before[n], p), \
                f"Encoder param {n} changed during training"

    def test_encoder_in_eval_mode(self, tiny_ae_seg_model, criterion,
                                   synthetic_loader, device):
        """train_one_epoch sets model.train() but encoder should be eval."""
        optimizer = torch.optim.SGD(
            tiny_ae_seg_model.seg_decoder.parameters(), lr=0.01)
        train_one_epoch(tiny_ae_seg_model, synthetic_loader, criterion,
                        optimizer, device)
        assert tiny_ae_seg_model.encoder.training is False

    def test_decoder_weights_update(self, tiny_ae_seg_model, criterion,
                                     synthetic_loader, device):
        """Decoder parameters should change after one training step."""
        decoder_before = {
            n: p.clone()
            for n, p in tiny_ae_seg_model.seg_decoder.named_parameters()
        }
        optimizer = torch.optim.SGD(
            tiny_ae_seg_model.seg_decoder.parameters(), lr=0.01)
        train_one_epoch(tiny_ae_seg_model, synthetic_loader, criterion,
                        optimizer, device)
        any_changed = any(
            not torch.equal(decoder_before[n], p)
            for n, p in tiny_ae_seg_model.seg_decoder.named_parameters()
        )
        assert any_changed

    def test_loss_decreases_over_epochs(self, tiny_ae_seg_model, criterion,
                                        synthetic_loader, device):
        """Loss should generally decrease over multiple epochs on fixed data."""
        optimizer = torch.optim.SGD(
            tiny_ae_seg_model.seg_decoder.parameters(), lr=0.01)
        losses = []
        for _ in range(5):
            loss = train_one_epoch(tiny_ae_seg_model, synthetic_loader,
                                   criterion, optimizer, device)
            losses.append(loss)
        assert losses[-1] < losses[0]


# ═══════════════════════════════════════════════
#  validate
# ═══════════════════════════════════════════════

class TestValidate:

    def test_returns_loss_and_metrics(self, tiny_ae_seg_model, criterion,
                                      synthetic_loader, device):
        avg_loss, metrics = validate(tiny_ae_seg_model, synthetic_loader,
                                     criterion, device)
        assert isinstance(avg_loss, float)
        assert avg_loss > 0

    def test_model_in_eval_mode(self, tiny_ae_seg_model, criterion,
                                 synthetic_loader, device):
        """validate() should set the model to eval mode."""
        validate(tiny_ae_seg_model, synthetic_loader, criterion, device)
        assert tiny_ae_seg_model.training is False

    def test_metrics_have_dice(self, tiny_ae_seg_model, criterion,
                                synthetic_loader, device):
        _, metrics = validate(tiny_ae_seg_model, synthetic_loader,
                              criterion, device)
        dice = metrics.mean_dice()
        assert isinstance(dice, float)
        assert 0.0 <= dice <= 1.0

    def test_metrics_have_iou(self, tiny_ae_seg_model, criterion,
                               synthetic_loader, device):
        _, metrics = validate(tiny_ae_seg_model, synthetic_loader,
                              criterion, device)
        iou = metrics.mean_iou()
        assert isinstance(iou, float)
        assert 0.0 <= iou <= 1.0

    def test_no_grad_context(self, tiny_ae_seg_model, criterion,
                              synthetic_loader, device):
        """Parameters should not accumulate gradients during validation."""
        validate(tiny_ae_seg_model, synthetic_loader, criterion, device)
        for param in tiny_ae_seg_model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)

    def test_confusion_matrix_shape(self, tiny_ae_seg_model, criterion,
                                     synthetic_loader, device):
        _, metrics = validate(tiny_ae_seg_model, synthetic_loader,
                              criterion, device)
        cm = metrics.confusion
        assert cm.shape == (NUM_CLASSES, NUM_CLASSES)

    def test_summary_keys(self, tiny_ae_seg_model, criterion,
                           synthetic_loader, device):
        _, metrics = validate(tiny_ae_seg_model, synthetic_loader,
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

    def _make_ae_checkpoint(self, path):
        """Create a fake autoencoder checkpoint on disk."""
        from task1.models import AEEncoder, AEDecoder
        encoder = AEEncoder(in_channels=IN_CH, features=FEATURES)
        torch.save({
            "epoch": 1,
            "model_state_dict": {},  # not needed — we only load encoder
            "encoder_state_dict": encoder.state_dict(),
            "best_loss": 0.5,
            "features": FEATURES,
            "args": {},
        }, path)

    def _make_loader(self):
        images = torch.randn(4, IN_CH, PATCH, PATCH)
        masks = torch.randint(0, NUM_CLASSES, (4, PATCH, PATCH))
        ds = TensorDataset(images, masks)
        return DataLoader(ds, batch_size=BATCH, shuffle=False)

    def test_end_to_end_training(self, tmp_path):
        """Run main() for 2 epochs on synthetic data, verify outputs."""
        from task1.train_ae_seg import main

        ckpt_dir = str(tmp_path / "checkpoints")
        results_dir = str(tmp_path / "results")
        os.makedirs(ckpt_dir, exist_ok=True)

        ae_ckpt = os.path.join(ckpt_dir, "autoencoder_best.pth")
        self._make_ae_checkpoint(ae_ckpt)

        loader = self._make_loader()

        fake_args = argparse.Namespace(
            ae_checkpoint=ae_ckpt,
            epochs=2,
            batch_size=BATCH,
            lr=1e-3,
            weight_decay=1e-4,
            patch_size=PATCH,
            features=FEATURES,
            data_root="fake",
            use_class_weights=False,
            checkpoint_dir=ckpt_dir,
            results_dir=results_dir,
            seed=42,
        )

        with patch("task1.train_ae_seg.parse_args",
                   return_value=fake_args), \
             patch("task1.train_ae_seg.get_segmentation_loaders",
                   return_value=(loader, loader, loader)), \
             patch("task1.train_ae_seg.get_device",
                   return_value=torch.device("cpu")):
            main()

        assert os.path.isfile(os.path.join(ckpt_dir, "ae_seg_best.pth"))
        assert os.path.isfile(
            os.path.join(results_dir, "ae_seg_training_curves.png"))
        assert os.path.isfile(
            os.path.join(results_dir, "ae_seg_val_predictions.png"))

    def test_checkpoint_contents(self, tmp_path):
        """Verify the saved checkpoint contains the expected keys."""
        from task1.train_ae_seg import main

        ckpt_dir = str(tmp_path / "checkpoints")
        results_dir = str(tmp_path / "results")
        os.makedirs(ckpt_dir, exist_ok=True)

        ae_ckpt = os.path.join(ckpt_dir, "autoencoder_best.pth")
        self._make_ae_checkpoint(ae_ckpt)

        loader = self._make_loader()

        fake_args = argparse.Namespace(
            ae_checkpoint=ae_ckpt,
            epochs=1,
            batch_size=BATCH,
            lr=1e-3,
            weight_decay=1e-4,
            patch_size=PATCH,
            features=FEATURES,
            data_root="fake",
            use_class_weights=False,
            checkpoint_dir=ckpt_dir,
            results_dir=results_dir,
            seed=42,
        )

        with patch("task1.train_ae_seg.parse_args",
                   return_value=fake_args), \
             patch("task1.train_ae_seg.get_segmentation_loaders",
                   return_value=(loader, loader, loader)), \
             patch("task1.train_ae_seg.get_device",
                   return_value=torch.device("cpu")):
            main()

        ckpt = torch.load(os.path.join(ckpt_dir, "ae_seg_best.pth"),
                          map_location="cpu", weights_only=False)
        assert "epoch" in ckpt
        assert "model_state_dict" in ckpt
        assert "best_dice" in ckpt
        assert "features" in ckpt
        assert "args" in ckpt
        assert isinstance(ckpt["best_dice"], float)
        assert ckpt["epoch"] >= 1

    def test_missing_ae_checkpoint_raises(self, tmp_path):
        """main() should raise FileNotFoundError if AE checkpoint missing."""
        from task1.train_ae_seg import main

        fake_args = argparse.Namespace(
            ae_checkpoint=str(tmp_path / "nonexistent.pth"),
            epochs=1,
            batch_size=BATCH,
            lr=1e-3,
            weight_decay=1e-4,
            patch_size=PATCH,
            features=FEATURES,
            data_root="fake",
            use_class_weights=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            results_dir=str(tmp_path / "results"),
            seed=42,
        )

        with patch("task1.train_ae_seg.parse_args",
                   return_value=fake_args), \
             patch("task1.train_ae_seg.get_device",
                   return_value=torch.device("cpu")):
            with pytest.raises(FileNotFoundError):
                main()

    def test_best_dice_positive(self, tmp_path):
        """After training, best_dice in the checkpoint should be > 0."""
        from task1.train_ae_seg import main

        ckpt_dir = str(tmp_path / "checkpoints")
        results_dir = str(tmp_path / "results")
        os.makedirs(ckpt_dir, exist_ok=True)

        ae_ckpt = os.path.join(ckpt_dir, "autoencoder_best.pth")
        self._make_ae_checkpoint(ae_ckpt)

        loader = self._make_loader()

        fake_args = argparse.Namespace(
            ae_checkpoint=ae_ckpt,
            epochs=2,
            batch_size=BATCH,
            lr=1e-3,
            weight_decay=1e-4,
            patch_size=PATCH,
            features=FEATURES,
            data_root="fake",
            use_class_weights=False,
            checkpoint_dir=ckpt_dir,
            results_dir=results_dir,
            seed=42,
        )

        with patch("task1.train_ae_seg.parse_args",
                   return_value=fake_args), \
             patch("task1.train_ae_seg.get_segmentation_loaders",
                   return_value=(loader, loader, loader)), \
             patch("task1.train_ae_seg.get_device",
                   return_value=torch.device("cpu")):
            main()

        ckpt = torch.load(os.path.join(ckpt_dir, "ae_seg_best.pth"),
                          map_location="cpu", weights_only=False)
        assert ckpt["best_dice"] > 0.0
