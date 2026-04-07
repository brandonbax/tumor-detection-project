"""
Task 2 — Test Set Evaluation

Runs both trained models on Task2_Test_Set (1858 pre-made 100x100 patches).
Labels are parsed from filenames, e.g.:
  test_set_metastatic_roi_013_nuclei_tumor_<uuid>.npy -> nuclei_tumor

Outputs: accuracy, precision, recall, F1 and confusion matrix for both approaches.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

TEST_DIR   = Path(__file__).parent.parent / "Task2_Test_Set"
CKPT_A     = Path(__file__).parent / "checkpoints_a" / "best_model.pth"
CKPT_B     = Path(__file__).parent / "checkpoints_b" / "best_model_b.pth"
ENCODER_B  = Path(__file__).parent / "checkpoints_b" / "supcon_encoder.pth"
OUTPUT_DIR = Path(__file__).parent / "test_results"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_CLASSES = ["nuclei_histiocyte", "nuclei_lymphocyte", "nuclei_tumor"]

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()          else
    torch.device("cpu")
)
print(f"Using device: {DEVICE}")

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class TestSetDataset(Dataset):
    def __init__(self, test_dir, class_to_idx, transform=None):
        self.transform = transform
        self.files, self.labels = [], []
        for path in sorted(test_dir.glob("*.npy")):
            for cls in class_to_idx:
                if cls in path.stem:
                    self.files.append(path)
                    self.labels.append(class_to_idx[cls])
                    break
        print(f"  Found {len(self.files)} test patches")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        patch = np.load(self.files[idx])
        if self.transform:
            patch = self.transform(patch)
        return patch, self.labels[idx]


def load_model_a(num_classes):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(CKPT_A, map_location=DEVICE))
    return model.to(DEVICE).eval()


def load_model_b(backbone_name, num_classes):
    # backbone_name can be e.g. "efficientnet_b0" or "efficientnet_b0_unfrozen"
    ckpt_dir   = Path(__file__).parent / f"checkpoints_b_{backbone_name}"
    model_ckpt = ckpt_dir / "best_model_b.pth"

    base = backbone_name.replace("_unfrozen", "").replace("_weighted", "")
    if base == "resnet18":
        m = models.resnet18(weights=None)
        encoder = nn.Sequential(*list(m.children())[:-1])
        feat_dim = 512
    elif base == "resnet50":
        m = models.resnet50(weights=None)
        encoder = nn.Sequential(*list(m.children())[:-1])
        feat_dim = 2048
    elif base == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        encoder = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1))
        feat_dim = 1280

    class EncoderClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder    = encoder
            self.classifier = nn.Linear(feat_dim, num_classes)

        def forward(self, x):
            return self.classifier(self.encoder(x).flatten(1))

    model = EncoderClassifier()
    model.load_state_dict(torch.load(model_ckpt, map_location=DEVICE))
    return model.to(DEVICE).eval()


@torch.no_grad()
def run_inference(model, loader, tta=False):
    """Run inference, optionally with test-time augmentation (TTA).
    TTA averages softmax probabilities over 4 views: original + hflip + vflip + both.
    """
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        if tta:
            logits  = model(imgs)
            logits += model(torch.flip(imgs, [3]))           # horizontal flip
            logits += model(torch.flip(imgs, [2]))           # vertical flip
            logits += model(torch.flip(imgs, [2, 3]))        # both
            preds   = logits.argmax(1).cpu().numpy()
        else:
            preds = model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(preds, labels, class_names, title, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[c.replace("nuclei_", "") for c in class_names],
                yticklabels=[c.replace("nuclei_", "") for c in class_names])
    plt.title(title); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    args = parser.parse_args()

    class_to_idx = {cls: i for i, cls in enumerate(TARGET_CLASSES)}
    loader = DataLoader(
        TestSetDataset(TEST_DIR, class_to_idx, val_transforms),
        batch_size=64, shuffle=False, num_workers=2, pin_memory=True
    )
    tta_suffix = "_tta" if args.tta else ""
    if args.tta:
        print("Test-time augmentation enabled (4 views: orig + hflip + vflip + both)")

    print("\n── Approach A (EfficientNet-B0, weighted loss) ──")
    preds_a, lbls = run_inference(load_model_a(len(TARGET_CLASSES)), loader, tta=args.tta)
    print(classification_report(lbls, preds_a, target_names=TARGET_CLASSES, digits=4))
    print(f"Overall accuracy: {accuracy_score(lbls, preds_a):.4f}")
    plot_confusion_matrix(preds_a, lbls, TARGET_CLASSES,
                          f"Approach A — Test Set{' (TTA)' if args.tta else ''}",
                          OUTPUT_DIR / f"confusion_matrix_a{tta_suffix}.png")

    for backbone in ["resnet18", "resnet50", "efficientnet_b0",
                     "efficientnet_b0_unfrozen", "efficientnet_b0_unfrozen_weighted",
                     "efficientnet_b0_unfrozen_all"]:
        ckpt = Path(__file__).parent / f"checkpoints_b_{backbone}" / "best_model_b.pth"
        if not ckpt.exists():
            print(f"\n── Approach B ({backbone}) — checkpoint not found, skipping ──")
            continue
        print(f"\n── Approach B SupCon ({backbone}){' + TTA' if args.tta else ''} ──")
        preds_b, _ = run_inference(load_model_b(backbone, len(TARGET_CLASSES)), loader, tta=args.tta)
        print(classification_report(lbls, preds_b, target_names=TARGET_CLASSES, digits=4))
        print(f"Overall accuracy: {accuracy_score(lbls, preds_b):.4f}")
        plot_confusion_matrix(preds_b, lbls, TARGET_CLASSES,
                              f"Approach B {backbone}{' (TTA)' if args.tta else ''} — Test Set",
                              OUTPUT_DIR / f"confusion_matrix_b_{backbone}{tta_suffix}.png")


if __name__ == "__main__":
    main()
