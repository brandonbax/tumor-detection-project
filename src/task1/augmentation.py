import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


def _build_seg_pipeline(size: int, is_train: bool) -> A.Compose:
    """
    Build an albumentations pipeline for segmentation.

    The pipeline is used via ``transform(image=..., mask=...)``;
    albumentations guarantees that every spatial transform is applied
    identically to both.
    """
    if is_train:
        return A.Compose([
            A.Resize(size, size),

            # Spatial
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                border_mode=0, p=0.3,
            ),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),

            # Pixel‑level (image only — masks are unaffected)
            A.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.05, p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(std_range=(0.01, 0.03), p=0.2),

            # Normalise + to tensor
            A.Normalize(mean=config.DATASET_MEAN, std=config.DATASET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=config.DATASET_MEAN, std=config.DATASET_STD),
            ToTensorV2(),
        ])


class SegmentationAugmentation:
    """
    Joint augmentation for (image, mask) pairs.

    Parameters
    ----------
    size : int
        Target spatial dimension (H = W).
    is_train : bool
        If False, only resize + normalise (no random transforms).
    """

    def __init__(self, size: int = config.PATCH_SIZE, is_train: bool = True):
        self.pipeline = _build_seg_pipeline(size, is_train)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        image : np.ndarray  (H, W, 3), uint8
        mask  : np.ndarray  (H, W),    uint8

        Returns
        -------
        image_tensor : torch.Tensor  (3, size, size), float32
        mask_tensor  : torch.Tensor  (size, size),    int64
        """
        result = self.pipeline(image=image, mask=mask)
        image_t = result["image"]                   # (3, H, W) float32
        mask_t = result["mask"].long()              # (H, W) int64
        return image_t, mask_t


def _build_image_pipeline(size: int, is_train: bool) -> A.Compose:
    """
    Build an albumentations pipeline for image-only transforms
    (no mask argument needed).
    """
    if is_train:
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.05, p=0.5,
            ),
            A.Normalize(mean=config.DATASET_MEAN, std=config.DATASET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=config.DATASET_MEAN, std=config.DATASET_STD),
            ToTensorV2(),
        ])


class ImageTransform:
    """
    Callable image-only transform (wraps an albumentations pipeline).

    Accepts a numpy array (H, W, 3) uint8 and returns a float tensor.
    """

    def __init__(self, size: int = config.PATCH_SIZE, is_train: bool = True):
        self.pipeline = _build_image_pipeline(size, is_train)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        return self.pipeline(image=image)["image"]


def get_image_transforms(patch_size: int = config.PATCH_SIZE,
                         is_train: bool = True) -> ImageTransform:
    """
    Return an image-only transform for autoencoder / unlabelled data.
    """
    return ImageTransform(size=patch_size, is_train=is_train)