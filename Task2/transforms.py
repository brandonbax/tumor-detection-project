"""
augmentation.py
---------------
Transform pipelines used across Task 2 training and evaluation.

  train_transforms  — standard augmentation for supervised fine-tuning
  val_transforms    — deterministic normalisation only (eval / test)
  supcon_transforms — aggressive augmentation for contrastive pre-training
"""
from torchvision import transforms
import config

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])

# More aggressive augmentation for contrastive pre-training — each patch is
# augmented twice independently to create positive pairs
supcon_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(size=96, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=9)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])
