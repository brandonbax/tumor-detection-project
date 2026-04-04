"""
Task 2 — Test Set Evaluation

Runs both trained models on Task2_Test_Set (1858 pre-made 100x100 patches).
Labels are parsed from filenames, e.g.:
  test_set_metastatic_roi_013_nuclei_tumor_<uuid>.npy -> nuclei_tumor

Outputs: accuracy, precision, recall, F1, confusion matrix for both approaches.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
TEST_DIR    = Path(__file__).parent.parent / "Task2_Test_Set"
DATA_DIR    = Path(__file__).parent / "task2_dataset"
CKPT_A      = Path(__file__).parent / "approach_a_improved" / "checkpoints_a_weighted" / "best_model.pth"
CKPT_A_ORIG = Path(__file__).parent / "checkpoints_a" / "best_model.pth"
CKPT_B      = Path(__file__).parent / "checkpoints_b" / "best_model_b.pth"
ENCODER_B   = Path(__file__).parent / "checkpoints_b" / "simclr_encoder.pth"
OUTPUT_DIR  = Path(__file__).parent / "test_results"
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


# ── Dataset ────────────────────────────────────────────────────────────────────
class TestSetDataset(Dataset):
    def __init__(self, test_dir, class_to_idx, transform=None):
        self.transform    = transform
        self.class_to_idx = class_to_idx
        self.files, self.labels = [], []

        for path in sorted(test_dir.glob("*.npy")):
            # Parse class from filename — class name is the second-to-last underscore segment group
            for cls in class_to_idx:
                if cls in path.stem:
                    self.files.append(path)
                    self.labels.append(class_to_idx[cls])
                    break

        print(f"  Found {len(self.files)} test patches from {test_dir.name}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        patch = np.load(self.files[idx])
        label = self.labels[idx]
        if self.transform:
            patch = self.transform(patch)
        return patch, label


# ── Models ─────────────────────────────────────────────────────────────────────
def load_model_a(num_classes, class_to_idx):
    model    = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(CKPT_A, map_location=DEVICE))
    return model.to(DEVICE).eval()


def load_model_b(num_classes):
    backbone = models.resnet18(weights=None)
    encoder  = nn.Sequential(*list(backbone.children())[:-1])
    encoder.load_state_dict(torch.load(ENCODER_B, map_location=DEVICE))
    for p in encoder.parameters():
        p.requires_grad = False

    class FrozenEncoderClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder    = encoder
            self.classifier = nn.Linear(512, num_classes)

        def forward(self, x):
            with torch.no_grad():
                f = self.encoder(x).flatten(1)
            return self.classifier(f)

    model = FrozenEncoderClassifier()
    model.load_state_dict(torch.load(CKPT_B, map_location=DEVICE))
    return model.to(DEVICE).eval()


# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader):
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        preds  = model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


# ── Confusion Matrix ───────────────────────────────────────────────────────────
def plot_confusion_matrix(preds, labels, class_names, title, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[c.replace("nuclei_", "") for c in class_names],
                yticklabels=[c.replace("nuclei_", "") for c in class_names])
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    class_to_idx = {cls: i for i, cls in enumerate(TARGET_CLASSES)}
    class_names  = TARGET_CLASSES

    dataset = TestSetDataset(TEST_DIR, class_to_idx, transform=val_transforms)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         num_workers=2, pin_memory=False)

    # ── Approach A original ────────────────────────────────────────────────────
    print("\n── Approach A original (ResNet-18, no class weights) ──")
    global CKPT_A
    CKPT_A        = CKPT_A_ORIG
    model_a       = load_model_a(len(TARGET_CLASSES), class_to_idx)
    preds_a, lbls = run_inference(model_a, loader)
    print(classification_report(lbls, preds_a, target_names=class_names, digits=4))
    print(f"Overall accuracy: {accuracy_score(lbls, preds_a):.4f}")
    plot_confusion_matrix(preds_a, lbls, class_names,
                          "Approach A — Confusion Matrix (Test Set)",
                          OUTPUT_DIR / "confusion_matrix_a.png")

    # ── Approach A weighted 2x ─────────────────────────────────────────────────
    CKPT_A_W = Path(__file__).parent / "approach_a_improved" / "checkpoints_a_weighted" / "best_model.pth"
    if CKPT_A_W.exists():
        print("\n── Approach A Weighted (histiocyte weight=2.0) ──")
        CKPT_A      = CKPT_A_W
        model_aw    = load_model_a(len(TARGET_CLASSES), class_to_idx)
        preds_aw, _ = run_inference(model_aw, loader)
        print(classification_report(lbls, preds_aw, target_names=class_names, digits=4))
        print(f"Overall accuracy: {accuracy_score(lbls, preds_aw):.4f}")
        plot_confusion_matrix(preds_aw, lbls, class_names,
                              "Approach A Weighted 2x — Confusion Matrix (Test Set)",
                              OUTPUT_DIR / "confusion_matrix_a_weighted2.png")

    # ── Approach A weighted 3x ─────────────────────────────────────────────────
    CKPT_A_W3 = Path(__file__).parent / "approach_a_improved" / "checkpoints_a_weighted3" / "best_model.pth"
    if CKPT_A_W3.exists():
        print("\n── Approach A Weighted (histiocyte weight=3.0) ──")
        CKPT_A       = CKPT_A_W3
        model_aw3    = load_model_a(len(TARGET_CLASSES), class_to_idx)
        preds_aw3, _ = run_inference(model_aw3, loader)
        print(classification_report(lbls, preds_aw3, target_names=class_names, digits=4))
        print(f"Overall accuracy: {accuracy_score(lbls, preds_aw3):.4f}")
        plot_confusion_matrix(preds_aw3, lbls, class_names,
                              "Approach A Weighted 3x — Confusion Matrix (Test Set)",
                              OUTPUT_DIR / "confusion_matrix_a_weighted3.png")

    # ── Approach B ─────────────────────────────────────────────────────────────
    print("\n── Approach B (SimCLR + Linear Head) ──")
    model_b       = load_model_b(len(TARGET_CLASSES))
    preds_b, _    = run_inference(model_b, loader)
    print(classification_report(lbls, preds_b, target_names=class_names, digits=4))
    print(f"Overall accuracy: {accuracy_score(lbls, preds_b):.4f}")
    plot_confusion_matrix(preds_b, lbls, class_names,
                          "Approach B — Confusion Matrix (Test Set)",
                          OUTPUT_DIR / "confusion_matrix_b.png")


if __name__ == "__main__":
    main()
