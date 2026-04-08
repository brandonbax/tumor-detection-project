import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from task1.dataset import TissueSegmentationDataset

@pytest.fixture
def dummy_dataset_paths():
    """Returns dummy paths for testing."""
    return "dummy_images", "dummy_masks"

def test_tissue_segmentation_dataset_length(mocker, dummy_dataset_paths):
    """Test that the dataset returns the correct length."""
    mock_match = mocker.patch('task1.dataset.match_image_label_pairs')
    mocker.patch('task1.dataset.create_mask')

    image_dir, mask_dir = dummy_dataset_paths
    mock_match.return_value = [("img0.tif", "lbl0.geojson"), ("img1.tif", "lbl1.geojson"), ("img2.tif", "lbl2.geojson")]

    dataset = TissueSegmentationDataset(image_dir, mask_dir)
    assert len(dataset) == 3

def test_tissue_segmentation_dataset_getitem(mocker, dummy_dataset_paths):
    """Test that the dataset returns items with correct shapes and types."""
    mock_match = mocker.patch('task1.dataset.match_image_label_pairs')
    mock_read_tif = mocker.patch('task1.dataset.read_tif_image')
    mock_create_mask = mocker.patch('task1.dataset.create_mask')
    
    image_dir, mask_dir = dummy_dataset_paths
    mock_match.return_value = [("img0.tif", "lbl0.geojson")]
    
    # Mock return values for the dependencies
    mock_read_tif.return_value = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    mock_create_mask.return_value = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
    
    # Use patch_size=256 which is default config generally
    dataset = TissueSegmentationDataset(image_dir, mask_dir, patch_size=256)
    
    image, mask = dataset[0]
    
    # image should be a torch tensor after augmentations
    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.float32
    assert image.shape == (3, 256, 256)
    
    # mask should be a torch LongTensor
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.long
    assert mask.shape == (256, 256)

def test_tissue_segmentation_dataset_eval_mode(mocker, dummy_dataset_paths):
    """Test that the dataset returns items correctly in non-train mode."""
    mock_match = mocker.patch('task1.dataset.match_image_label_pairs')
    mock_read_tif = mocker.patch('task1.dataset.read_tif_image')
    mock_create_mask = mocker.patch('task1.dataset.create_mask')
    
    image_dir, mask_dir = dummy_dataset_paths
    mock_match.return_value = [("img0.tif", "lbl0.geojson")]
    
    mock_read_tif.return_value = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    mock_create_mask.return_value = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    
    dataset = TissueSegmentationDataset(image_dir, mask_dir, patch_size=128, is_train=False)
    
    image, mask = dataset[0]
    
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 128, 128)
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (128, 128)

def test_tissue_segmentation_dataset_empty_directory(mocker, dummy_dataset_paths):
    """Test dataset behavior with empty directory."""
    mock_match = mocker.patch('task1.dataset.match_image_label_pairs')
    
    image_dir, mask_dir = dummy_dataset_paths
    mock_match.return_value = []
    
    dataset = TissueSegmentationDataset(image_dir, mask_dir)
    assert len(dataset) == 0
