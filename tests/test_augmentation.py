import numpy as np
import pytest
import torch
import albumentations as A

from task1.augmentation import (
    _build_seg_pipeline,
    SegmentationAugmentation,
    _build_image_pipeline,
    ImageTransform,
    get_image_transforms
)
import config

@pytest.fixture
def dummy_image():
    """Create a dummy RGB image of shape (300, 400, 3)"""
    return np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)

@pytest.fixture
def dummy_mask():
    """Create a dummy segmentation mask of shape (300, 400)"""
    return np.random.randint(0, 3, (300, 400), dtype=np.uint8)

def test_build_seg_pipeline_train():
    """Test building a segmentation pipeline for training."""
    pipeline = _build_seg_pipeline(size=256, is_train=True)
    assert isinstance(pipeline, A.Compose)
    # Check that it contains expected components (simplistic check)
    has_resize = any(isinstance(t, A.Resize) for t in pipeline.transforms)
    has_norm = any(isinstance(t, A.Normalize) for t in pipeline.transforms)
    assert has_resize
    assert has_norm

def test_build_seg_pipeline_val():
    """Test building a segmentation pipeline for validation."""
    pipeline = _build_seg_pipeline(size=128, is_train=False)
    assert isinstance(pipeline, A.Compose)
    # It should have fewer transforms than the train pipeline
    train_pipeline = _build_seg_pipeline(size=128, is_train=True)
    assert len(pipeline.transforms) < len(train_pipeline.transforms)

def test_segmentation_augmentation_call(dummy_image, dummy_mask):
    """Test the __call__ method of SegmentationAugmentation."""
    aug = SegmentationAugmentation(size=224, is_train=False)
    
    image_t, mask_t = aug(dummy_image, dummy_mask)
    
    # Check output types
    assert isinstance(image_t, torch.Tensor)
    assert isinstance(mask_t, torch.Tensor)
    
    # Check dtypes
    assert image_t.dtype == torch.float32
    assert mask_t.dtype == torch.long
    
    # Check shapes
    assert image_t.shape == (3, 224, 224)
    assert mask_t.shape == (224, 224)

def test_segmentation_augmentation_train_call(dummy_image, dummy_mask):
    """Test the __call__ method of SegmentationAugmentation in train mode."""
    aug = SegmentationAugmentation(size=256, is_train=True)
    
    image_t, mask_t = aug(dummy_image, dummy_mask)
    assert image_t.shape == (3, 256, 256)
    assert mask_t.shape == (256, 256)

def test_build_image_pipeline():
    """Test building an image-only pipeline."""
    pipeline = _build_image_pipeline(size=128, is_train=True)
    assert isinstance(pipeline, A.Compose)
    
    val_pipeline = _build_image_pipeline(size=128, is_train=False)
    assert len(val_pipeline.transforms) < len(pipeline.transforms)

def test_image_transform_call(dummy_image):
    """Test the __call__ method of ImageTransform."""
    transform = ImageTransform(size=112, is_train=False)
    
    image_t = transform(dummy_image)
    
    assert isinstance(image_t, torch.Tensor)
    assert image_t.dtype == torch.float32
    assert image_t.shape == (3, 112, 112)

def test_get_image_transforms():
    """Test the get_image_transforms factory function."""
    transform = get_image_transforms(patch_size=320, is_train=True)
    assert isinstance(transform, ImageTransform)
    # The internal pipeline should reflect the requested size
    # We can indirectly verify this by running dummy data
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    image_t = transform(dummy_img)
    assert image_t.shape == (3, 320, 320)
