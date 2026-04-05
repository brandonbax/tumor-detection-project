import os
import glob

from torch.utils.data import Dataset, DataLoader

import config
from task1.data_processing import read_tif_image, create_mask, match_image_label_pairs
from task1.augmentation import SegmentationAugmentation, get_image_transforms


class TissueSegmentationDataset(Dataset):
    """
    Labelled dataset for tissue segmentation.

    Each sample is an (image, mask) pair where the mask has integer
    label.
    """
    def __init__(self,
                 image_dir: str,
                 label_dir: str,
                 patch_size: int = config.PATCH_SIZE,
                 is_train: bool = True,
                 class_map: dict = None):
        super().__init__()
        self.pairs = match_image_label_pairs(image_dir, label_dir)
        self.class_map = class_map or config.CLASS_NAME_TO_ID
        self.augment = SegmentationAugmentation(size=patch_size,
                                                is_train=is_train)

        # Pre-generate all masks in the main process so that
        # dataloader workers only ever read cached files.
        # (rasterio writes are not fork-safe for num_workers > 0)
        os.makedirs(config.MASK_DIR, exist_ok=True)
        for img_path, lbl_path in self.pairs:
            mask_path = os.path.join(config.MASK_DIR,
                                     os.path.basename(img_path))
            if not os.path.exists(mask_path):
                create_mask(lbl_path, img_path, mask_path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]

        image = read_tif_image(img_path)
        output_mask_path = os.path.join(config.MASK_DIR, os.path.basename(img_path))
        mask = create_mask(lbl_path, img_path, output_mask_path)

        image_t, mask_t = self.augment(image, mask)
        return image_t, mask_t


class UnlabelledImageDataset(Dataset):
    """
    Unlabelled dataset that returns only images.
    Used for autoencoder reconstruction pre-training.
    """

    def __init__(self,
                 image_dir: str,
                 patch_size: int = config.PATCH_SIZE,
                 is_train: bool = True):
        super().__init__()
        exts = (".tif", ".tiff", ".png", ".jpg")
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(exts)
        ])
        if not self.image_paths:
            # Fallback: recursive glob
            self.image_paths = sorted(
                glob.glob(os.path.join(image_dir, "**", "*.tif"),
                           recursive=True))

        self.transform = get_image_transforms(patch_size, is_train)

        print(f"[INFO] UnlabelledImageDataset: "
              f"{len(self.image_paths)} images from {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = read_tif_image(self.image_paths[idx])
        return self.transform(image)


def get_segmentation_loaders(batch_size: int = config.UNET_BATCH_SIZE,
                             num_workers: int = config.NUM_WORKERS,
                             patch_size: int = config.PATCH_SIZE):
    """
    Returns (train_loader, val_loader, test_loader) for tissue
    segmentation.
    """
    train_ds = TissueSegmentationDataset(
        config.TRAIN_IMAGE_DIR, config.TRAIN_LABEL_DIR,
        patch_size=patch_size, is_train=True)
    val_ds = TissueSegmentationDataset(
        config.VAL_IMAGE_DIR, config.VAL_LABEL_DIR,
        patch_size=patch_size, is_train=False)
    test_ds = TissueSegmentationDataset(
        config.TEST_IMAGE_DIR, config.TEST_LABEL_DIR,
        patch_size=patch_size, is_train=False)

    kw = dict(num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, **kw)
    return train_loader, val_loader, test_loader


def get_unlabelled_loaders(batch_size: int = config.AE_BATCH_SIZE,
                           num_workers: int = config.NUM_WORKERS,
                           patch_size: int = config.PATCH_SIZE):
    """
    Returns (train_loader, val_loader) of unlabelled images for
    autoencoder pre-training.
    """
    train_ds = UnlabelledImageDataset(
        config.TRAIN_IMAGE_DIR, patch_size=patch_size, is_train=True)
    val_ds = UnlabelledImageDataset(
        config.VAL_IMAGE_DIR, patch_size=patch_size, is_train=False)

    kw = dict(num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, **kw)
    return train_loader, val_loader