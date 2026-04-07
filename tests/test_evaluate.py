"""
test_evaluate.py
----------------
Unit tests for src/task1/evaluate.py

Covers:
  - parse_args       (CLI argument parsing and defaults)
  - evaluate_model   (metrics collection, prediction grid & confusion matrix output)
  - load_unet        (checkpoint → UNet model)
  - load_ae_seg      (checkpoint → AESegmentationModel)
  - main             (end-to-end with mocked loaders and checkpoints)
"""

import os
import json
import argparse

import torch
import pytest
from unittest.mock import patch
from torch.utils.data import DataLoader, TensorDataset

import config
from task1.evaluate import (
    parse_args, evaluate_model, load_unet, load_ae_seg,
)


# ── Constants ────────────────────────────────────────

NUM_CLASSES = 3
PATCH = 16
BATCH = 2
IN_CH = 3
UNET_FEATURES = [8, 16, 32]
AE_FEATURES = [8, 16, 32]


# ── Helpers ──────────────────────────────────────────

def _make_unet_checkpoint(path, features=None):
    """Create a valid UNet checkpoint on disk."""
    from task1.models import UNet
    if features is None:
        features = UNET_FEATURES
    model = UNet(in_channels=IN_CH, num_classes=NUM_CLASSES, features=features)
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "best_dice": 0.5,
        "args": {"features": features},
    }, path)


def _make_ae_seg_checkpoint(path, features=None):
    """Create a valid AE-Seg checkpoint on disk."""
    from task1.models import AEEncoder, AESegmentationModel
    if features is None:
        features = AE_FEATURES
    encoder = AEEncoder(in_channels=IN_CH, features=features)
    model = AESegmentationModel(encoder, num_classes=NUM_CLASSES,
                                features=features, freeze_encoder=True)
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "best_dice": 0.4,
        "features": features,
    }, path)


def _make_loader():
    images = torch.randn(4, IN_CH, PATCH, PATCH)
    masks = torch.randint(0, NUM_CLASSES, (4, PATCH, PATCH))
    ds = TensorDataset(images, masks)
    return DataLoader(ds, batch_size=BATCH, shuffle=False)


# ── Fixtures ─────────────────────────────────────────

@pytest.fixture
def synthetic_loader():
    return _make_loader()


@pytest.fixture
def device():
    return torch.device("cpu")


# ═══════════════════════════════════════════════
#  parse_args
# ═══════════════════════════════════════════════

class TestParseArgs:

    def test_defaults(self):
        with patch("sys.argv", ["evaluate.py"]):
            args = parse_args()
        assert args.unet_ckpt == os.path.join(
            config.CHECKPOINT_DIR, "unet_best.pth")
        assert args.ae_seg_ckpt == os.path.join(
            config.CHECKPOINT_DIR, "ae_seg_best.pth")
        assert args.batch_size == config.UNET_BATCH_SIZE
        assert args.patch_size == config.PATCH_SIZE
        assert args.data_root == config.DATASET_ROOT
        assert args.results_dir == config.RESULTS_DIR
        assert args.seed == config.SEED

    def test_custom_unet_ckpt(self):
        with patch("sys.argv",
                   ["evaluate.py", "--unet_ckpt", "/tmp/unet.pth"]):
            args = parse_args()
        assert args.unet_ckpt == "/tmp/unet.pth"

    def test_custom_ae_seg_ckpt(self):
        with patch("sys.argv",
                   ["evaluate.py", "--ae_seg_ckpt", "/tmp/ae_seg.pth"]):
            args = parse_args()
        assert args.ae_seg_ckpt == "/tmp/ae_seg.pth"

    def test_custom_batch_size(self):
        with patch("sys.argv", ["evaluate.py", "--batch_size", "4"]):
            args = parse_args()
        assert args.batch_size == 4

    def test_custom_data_root(self):
        with patch("sys.argv",
                   ["evaluate.py", "--data_root", "/tmp/mydata"]):
            args = parse_args()
        assert args.data_root == "/tmp/mydata"

    def test_custom_seed(self):
        with patch("sys.argv", ["evaluate.py", "--seed", "99"]):
            args = parse_args()
        assert args.seed == 99


# ═══════════════════════════════════════════════
#  evaluate_model
# ═══════════════════════════════════════════════

class TestEvaluateModel:

    def test_returns_metrics(self, synthetic_loader, device, tmp_path):
        from task1.models import UNet
        model = UNet(in_channels=IN_CH, num_classes=NUM_CLASSES,
                     features=UNET_FEATURES).to(device)
        results_dir = str(tmp_path / "results")
        os.makedirs(results_dir, exist_ok=True)

        metrics = evaluate_model(model, synthetic_loader, device,
                                 "test_model", results_dir)
        dice = metrics.mean_dice()
        assert isinstance(dice, float)
        assert 0.0 <= dice <= 1.0

    def test_model_in_eval_mode(self, synthetic_loader, device, tmp_path):
        from task1.models import UNet
        model = UNet(in_channels=IN_CH, num_classes=NUM_CLASSES,
                     features=UNET_FEATURES).to(device)
        model.train()
        results_dir = str(tmp_path / "results")
        os.makedirs(results_dir, exist_ok=True)

        evaluate_model(model, synthetic_loader, device,
                       "test_model", results_dir)
        assert model.training is False

    def test_saves_prediction_grid(self, synthetic_loader, device, tmp_path):
        from task1.models import UNet
        model = UNet(in_channels=IN_CH, num_classes=NUM_CLASSES,
                     features=UNET_FEATURES).to(device)
        results_dir = str(tmp_path / "results")
        os.makedirs(results_dir, exist_ok=True)

        evaluate_model(model, synthetic_loader, device,
                       "mymodel", results_dir)
        assert os.path.isfile(
            os.path.join(results_dir, "mymodel_test_predictions.png"))

    def test_saves_confusion_matrix(self, synthetic_loader, device, tmp_path):
        from task1.models import UNet
        model = UNet(in_channels=IN_CH, num_classes=NUM_CLASSES,
                     features=UNET_FEATURES).to(device)
        results_dir = str(tmp_path / "results")
        os.makedirs(results_dir, exist_ok=True)

        evaluate_model(model, synthetic_loader, device,
                       "mymodel", results_dir)
        assert os.path.isfile(
            os.path.join(results_dir, "mymodel_confusion_matrix.png"))

    def test_no_grad_context(self, synthetic_loader, device, tmp_path):
        from task1.models import UNet
        model = UNet(in_channels=IN_CH, num_classes=NUM_CLASSES,
                     features=UNET_FEATURES).to(device)
        results_dir = str(tmp_path / "results")
        os.makedirs(results_dir, exist_ok=True)

        evaluate_model(model, synthetic_loader, device,
                       "test_model", results_dir)
        for param in model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)

    def test_metrics_summary_keys(self, synthetic_loader, device, tmp_path):
        from task1.models import UNet
        model = UNet(in_channels=IN_CH, num_classes=NUM_CLASSES,
                     features=UNET_FEATURES).to(device)
        results_dir = str(tmp_path / "results")
        os.makedirs(results_dir, exist_ok=True)

        metrics = evaluate_model(model, synthetic_loader, device,
                                 "test_model", results_dir)
        summary = metrics.summary()
        assert "pixel_accuracy" in summary
        assert "mean_dice" in summary
        assert "mean_iou" in summary


# ═══════════════════════════════════════════════
#  load_unet
# ═══════════════════════════════════════════════

class TestLoadUnet:

    def test_loads_model(self, tmp_path, device):
        ckpt_path = str(tmp_path / "unet.pth")
        _make_unet_checkpoint(ckpt_path)
        model = load_unet(ckpt_path, device)
        assert model is not None
        # Should be able to do a forward pass
        x = torch.randn(1, IN_CH, PATCH, PATCH, device=device)
        out = model(x)
        assert out.shape == (1, NUM_CLASSES, PATCH, PATCH)

    def test_uses_features_from_checkpoint(self, tmp_path, device):
        ckpt_path = str(tmp_path / "unet.pth")
        features = [16, 32, 64]
        _make_unet_checkpoint(ckpt_path, features=features)
        model = load_unet(ckpt_path, device)
        x = torch.randn(1, IN_CH, PATCH, PATCH, device=device)
        out = model(x)
        assert out.shape == (1, NUM_CLASSES, PATCH, PATCH)

    def test_raises_on_missing_file(self, device):
        with pytest.raises(Exception):
            load_unet("/nonexistent/path.pth", device)


# ═══════════════════════════════════════════════
#  load_ae_seg
# ═══════════════════════════════════════════════

class TestLoadAeSeg:

    def test_loads_model(self, tmp_path, device):
        ckpt_path = str(tmp_path / "ae_seg.pth")
        _make_ae_seg_checkpoint(ckpt_path)
        model = load_ae_seg(ckpt_path, device)
        assert model is not None
        x = torch.randn(1, IN_CH, PATCH, PATCH, device=device)
        out = model(x)
        assert out.shape == (1, NUM_CLASSES, PATCH, PATCH)

    def test_encoder_is_frozen(self, tmp_path, device):
        ckpt_path = str(tmp_path / "ae_seg.pth")
        _make_ae_seg_checkpoint(ckpt_path)
        model = load_ae_seg(ckpt_path, device)
        for param in model.encoder.parameters():
            assert param.requires_grad is False

    def test_uses_features_from_checkpoint(self, tmp_path, device):
        ckpt_path = str(tmp_path / "ae_seg.pth")
        features = [16, 32, 64]
        _make_ae_seg_checkpoint(ckpt_path, features=features)
        model = load_ae_seg(ckpt_path, device)
        x = torch.randn(1, IN_CH, PATCH, PATCH, device=device)
        out = model(x)
        assert out.shape == (1, NUM_CLASSES, PATCH, PATCH)

    def test_raises_on_missing_file(self, device):
        with pytest.raises(Exception):
            load_ae_seg("/nonexistent/path.pth", device)


# ═══════════════════════════════════════════════
#  main (integration, end-to-end with mocks)
# ═══════════════════════════════════════════════

class TestMain:

    def test_both_models(self, tmp_path):
        """Run main() with both checkpoints present, verify all outputs."""
        from task1.evaluate import main

        ckpt_dir = str(tmp_path / "checkpoints")
        results_dir = str(tmp_path / "results")
        os.makedirs(ckpt_dir, exist_ok=True)

        unet_ckpt = os.path.join(ckpt_dir, "unet_best.pth")
        ae_seg_ckpt = os.path.join(ckpt_dir, "ae_seg_best.pth")
        _make_unet_checkpoint(unet_ckpt)
        _make_ae_seg_checkpoint(ae_seg_ckpt)

        loader = _make_loader()

        fake_args = argparse.Namespace(
            unet_ckpt=unet_ckpt,
            ae_seg_ckpt=ae_seg_ckpt,
            batch_size=BATCH,
            patch_size=PATCH,
            data_root="fake",
            results_dir=results_dir,
            seed=42,
        )

        with patch("task1.evaluate.parse_args", return_value=fake_args), \
             patch("task1.evaluate.get_segmentation_loaders",
                   return_value=(loader, loader, loader)), \
             patch("task1.evaluate.get_device",
                   return_value=torch.device("cpu")), \
             patch("task1.evaluate.save_class_distribution",
                   return_value={}), \
             patch("task1.evaluate.TissueSegmentationDataset"):
            main()

        # UNet outputs
        assert os.path.isfile(
            os.path.join(results_dir, "unet_test_predictions.png"))
        assert os.path.isfile(
            os.path.join(results_dir, "unet_confusion_matrix.png"))

        # AE-Seg outputs
        assert os.path.isfile(
            os.path.join(results_dir, "ae_seg_test_predictions.png"))
        assert os.path.isfile(
            os.path.join(results_dir, "ae_seg_confusion_matrix.png"))

        # JSON results
        results_path = os.path.join(results_dir, "test_results.json")
        assert os.path.isfile(results_path)
        with open(results_path) as f:
            data = json.load(f)
        assert "UNet" in data
        assert "AE-Seg" in data
        assert "mean_dice" in data["UNet"]
        assert "mean_dice" in data["AE-Seg"]

    def test_unet_only(self, tmp_path):
        """Run main() with only UNet checkpoint; AE-Seg should be skipped."""
        from task1.evaluate import main

        ckpt_dir = str(tmp_path / "checkpoints")
        results_dir = str(tmp_path / "results")
        os.makedirs(ckpt_dir, exist_ok=True)

        unet_ckpt = os.path.join(ckpt_dir, "unet_best.pth")
        _make_unet_checkpoint(unet_ckpt)

        loader = _make_loader()

        fake_args = argparse.Namespace(
            unet_ckpt=unet_ckpt,
            ae_seg_ckpt=os.path.join(ckpt_dir, "nonexistent.pth"),
            batch_size=BATCH,
            patch_size=PATCH,
            data_root="fake",
            results_dir=results_dir,
            seed=42,
        )

        with patch("task1.evaluate.parse_args", return_value=fake_args), \
             patch("task1.evaluate.get_segmentation_loaders",
                   return_value=(loader, loader, loader)), \
             patch("task1.evaluate.get_device",
                   return_value=torch.device("cpu")), \
             patch("task1.evaluate.save_class_distribution",
                   return_value={}), \
             patch("task1.evaluate.TissueSegmentationDataset"):
            main()

        assert os.path.isfile(
            os.path.join(results_dir, "unet_test_predictions.png"))

        results_path = os.path.join(results_dir, "test_results.json")
        with open(results_path) as f:
            data = json.load(f)
        assert "UNet" in data
        assert "AE-Seg" not in data

    def test_no_checkpoints(self, tmp_path):
        """Run main() with no checkpoints — should still write empty JSON."""
        from task1.evaluate import main

        results_dir = str(tmp_path / "results")

        fake_args = argparse.Namespace(
            unet_ckpt=str(tmp_path / "no_unet.pth"),
            ae_seg_ckpt=str(tmp_path / "no_ae_seg.pth"),
            batch_size=BATCH,
            patch_size=PATCH,
            data_root="fake",
            results_dir=results_dir,
            seed=42,
        )

        with patch("task1.evaluate.parse_args", return_value=fake_args), \
             patch("task1.evaluate.get_segmentation_loaders",
                   return_value=(_make_loader(),) * 3), \
             patch("task1.evaluate.get_device",
                   return_value=torch.device("cpu")), \
             patch("task1.evaluate.save_class_distribution",
                   return_value={}), \
             patch("task1.evaluate.TissueSegmentationDataset"):
            main()

        results_path = os.path.join(results_dir, "test_results.json")
        assert os.path.isfile(results_path)
        with open(results_path) as f:
            data = json.load(f)
        assert data == {}

    def test_json_results_values_are_floats(self, tmp_path):
        """All metric values in the JSON should be plain floats."""
        from task1.evaluate import main

        ckpt_dir = str(tmp_path / "checkpoints")
        results_dir = str(tmp_path / "results")
        os.makedirs(ckpt_dir, exist_ok=True)

        unet_ckpt = os.path.join(ckpt_dir, "unet_best.pth")
        _make_unet_checkpoint(unet_ckpt)

        loader = _make_loader()

        fake_args = argparse.Namespace(
            unet_ckpt=unet_ckpt,
            ae_seg_ckpt=str(tmp_path / "nonexistent.pth"),
            batch_size=BATCH,
            patch_size=PATCH,
            data_root="fake",
            results_dir=results_dir,
            seed=42,
        )

        with patch("task1.evaluate.parse_args", return_value=fake_args), \
             patch("task1.evaluate.get_segmentation_loaders",
                   return_value=(loader, loader, loader)), \
             patch("task1.evaluate.get_device",
                   return_value=torch.device("cpu")), \
             patch("task1.evaluate.save_class_distribution",
                   return_value={}), \
             patch("task1.evaluate.TissueSegmentationDataset"):
            main()

        results_path = os.path.join(results_dir, "test_results.json")
        with open(results_path) as f:
            data = json.load(f)

        for key, val in data["UNet"].items():
            assert isinstance(val, (int, float)), \
                f"UNet[{key!r}] is {type(val).__name__}, expected numeric"
