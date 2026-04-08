"""
patches.py - dataset classes for Task 2 nuclei patches.
"""
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import config


class NucleiDataset(Dataset):
    """Balanced nuclei patch dataset loaded from .npy arrays."""

    def __init__(self, split: str, transform=None):
        self.X         = np.load(config.DATA_DIR / f"X_{split}.npy")
        self.y         = np.load(config.DATA_DIR / f"y_{split}.npy")
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img, label = self.X[idx], int(self.y[idx])
        if self.transform:
            img = self.transform(img)
        return img, label


class SupConDataset(Dataset):
    """Returns two independently augmented views of each patch for SupCon."""

    def __init__(self, transform):
        self.X         = np.load(config.DATA_DIR / "X_contrastive.npy")
        self.y         = np.load(config.DATA_DIR / "y_contrastive.npy")
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx]
        # Two independent random augmentations form a positive pair
        return self.transform(img), self.transform(img), int(self.y[idx])


class TestSetDataset(Dataset):
    """Pre-made 100×100 test patches; class label inferred from filename."""

    def __init__(self, transform=None):
        self.transform = transform
        class_to_idx   = {c: i for i, c in enumerate(config.TARGET_CLASSES)}
        self.files, self.labels = [], []
        for path in sorted(config.TEST_DIR.glob("*.npy")):
            for cls, idx in class_to_idx.items():
                if cls in path.stem:
                    self.files.append(path)
                    self.labels.append(idx)
                    break
        print(f"  Found {len(self.files)} test patches")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        patch = np.load(self.files[idx])
        if self.transform:
            patch = self.transform(patch)
        return patch, self.labels[idx]


def get_loaders(batch_size: int, train_transform, val_transform):
    """Return (train_loader, val_loader) for the balanced nuclei splits."""
    train_ds = NucleiDataset("train",      transform=train_transform)
    val_ds   = NucleiDataset("validation", transform=val_transform)
    kw = dict(num_workers=2, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
    )
