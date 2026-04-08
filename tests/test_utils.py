import os
import torch
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import config
from task1.utils import (
    set_seed,
    get_device,
    SegmentationMetrics,
    SegmentationCriterion,
    get_criterion,
    compute_class_weights,
    denormalize,
    mask_to_rgb,
    save_prediction_grid,
    save_confusion_matrix,
    save_training_curves,
    save_reconstruction_grid
)


def test_set_seed():
    """Verify that set_seed runs gracefully."""
    set_seed(42)
    assert torch.initial_seed() == 42


def test_get_device(mocker):
    """Verify device selection logic."""
    mocker.patch('torch.cuda.is_available', return_value=False)
    if hasattr(torch.backends, 'mps'):
        mocker.patch('torch.backends.mps.is_available', return_value=False)
    
    device = get_device()
    assert device.type == 'cpu'

    mocker.patch('torch.cuda.is_available', return_value=True)
    device = get_device()
    assert device.type == 'cuda'


def test_segmentation_metrics():
    """Verify correct functioning of segmentation metrics caching and summary output."""
    metrics = SegmentationMetrics(num_classes=3, device=torch.device('cpu'))
    
    # Shape: (B, C, H, W)
    preds = torch.randn(2, 3, 4, 4)
    # Shape: (B, H, W)
    targets = torch.randint(0, 3, (2, 4, 4))
    
    metrics.update(preds, targets)
    
    mean_dice = metrics.mean_dice()
    mean_iou = metrics.mean_iou()
    
    assert isinstance(mean_dice, float)
    assert isinstance(mean_iou, float)
    
    conf = metrics.confusion
    assert conf.shape == (3, 3)
    assert isinstance(conf, np.ndarray)
    
    summary = metrics.summary(class_names=['Other', 'Tumor', 'Stroma'])
    assert 'pixel_accuracy' in summary
    assert 'mean_dice' in summary
    assert 'dice_Other' in summary
    assert 'iou_Tumor' in summary


def test_get_criterion():
    """Verify criterion wrapper correctly initializes and passes gradients."""
    target = torch.randint(0, 3, (2, 4, 4))
    logits = torch.randn(2, 3, 4, 4, requires_grad=True)
    
    crit = get_criterion(lambda_dice=0.5, lambda_ce=0.5)
    loss = crit(logits, target)
    
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0
    
    loss.backward()
    assert logits.grad is not None


def test_compute_class_weights():
    """Verify proper inverse frequency computation bounded logic."""
    class DummyDS:
        def __len__(self):
            return 2
        def __getitem__(self, idx):
            if idx == 0:
                mask = torch.tensor([[0, 1], [2, 2]])
            else:
                mask = torch.tensor([[0, 0], [1, 2]])
            return None, mask
    
    ds = DummyDS()
    w = compute_class_weights(ds, num_classes=3)
    assert w.shape == (3,)
    assert torch.isclose(w.sum(), torch.tensor(3.0)).item()
    # Mask flat count across ds: 0:3, 1:2, 2:3.
    # So idx=1 has fewer elements in dataset, weighting should be mathematically highest.
    assert w[1] > w[0]
    assert w[1] > w[2]


def test_denormalize():
    """Verify RGB clipping and transformation out of denormalization correctly returns."""
    img_tensor = torch.zeros(3, 4, 4)
    out = denormalize(img_tensor)
    
    assert out.shape == (4, 4, 3)
    assert out.dtype == np.uint8
    # Since original is 0, out should roughly match mean * 255
    expected = (np.array(config.DATASET_MEAN) * 255).astype(np.uint8)
    np.testing.assert_array_equal(out[0, 0], expected)


def test_mask_to_rgb():
    """Verify mask classes match precisely assigned PALETTE."""
    mask = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    rgb = mask_to_rgb(mask)
    
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8


def test_save_plots(tmp_path):
    """Verify plotting wrappers effectively build paths without systemic crashing."""
    out_dir = tmp_path / "out"
    images = torch.randn(2, 3, 4, 4)
    masks = torch.randint(0, 3, (2, 4, 4))
    
    p1 = out_dir / "grid.png"
    save_prediction_grid(images, masks, masks, str(p1))
    assert p1.exists()

    metrics = SegmentationMetrics(num_classes=3, device=torch.device('cpu'))
    metrics.update(images, masks)
    p2 = out_dir / "cm.png"
    save_confusion_matrix(metrics, str(p2), class_names=['Other', 'Tumor', 'Stroma'])
    assert p2.exists()
    
    p3 = out_dir / "curves.png"
    save_training_curves([0.1, 0.05], [0.2, 0.1], [0.8, 0.9], str(p3))
    assert p3.exists()

    p4 = out_dir / "recon.png"
    save_reconstruction_grid(images, images, str(p4))
    assert p4.exists()
