# Task 2 — Nuclei Classification

Three-class patch classifier for H&E nuclei: `nuclei_histiocyte`, `nuclei_lymphocyte`, `nuclei_tumor`.

## Directory Structure

```
Task2/
├── dataset_preparation.py      # Step 0: extract and balance patches → task2_dataset/
├── pretrain_supcon.py          # Step 1 (Approach B): SupCon encoder pre-training
├── train_approach_a.py         # Train end-to-end EfficientNet-B0 classifier
├── train_approach_b.py         # Train linear head on frozen/unfrozen SupCon encoder
├── evaluate_test.py            # Evaluate all models on Task2_Test_Set/
├── task2_dataset/              # Balanced npy splits (gitignored)
├── checkpoints_a/              # Approach A checkpoint (gitignored)
├── checkpoints_b_<backbone>/   # SupCon encoder + linear head checkpoints (gitignored)
└── test_results/               # Confusion matrices + classification reports (gitignored)
```

## Setup

```bash
pip install torch torchvision scikit-learn matplotlib seaborn
```

## Workflow

### Step 0 — Prepare dataset

```bash
python Task2/dataset_preparation.py
```

Reads from `Dataset_Splits/train/` and `Dataset_Splits/validation/`, balances to 2500/700 per class, saves `task2_dataset/X_{train,validation}.npy` and `task2_dataset/y_{train,validation}.npy`.

### Approach A — End-to-end EfficientNet-B0

```bash
python Task2/train_approach_a.py
```

- EfficientNet-B0, weighted CrossEntropy (histiocyte weight=3.0)
- 50 epochs, early stopping (patience=7)
- Val accuracy: **0.7514** | Test accuracy: **0.7411**

### Approach B — SupCon pre-training + linear probe

**Step 1: Pre-train encoder with Supervised Contrastive Loss**

```bash
python Task2/pretrain_supcon.py --backbone efficientnet_b0
# also available: resnet18, resnet50
```

Saves `checkpoints_b_<backbone>/supcon_encoder.pth`.

**Step 2: Train linear classifier head**

```bash
# Frozen encoder (linear probe)
python Task2/train_approach_b.py --backbone efficientnet_b0

# Unfreeze last encoder block (features[7]+features[8]) with differential LR
python Task2/train_approach_b.py --backbone efficientnet_b0 --unfreeze

# Unfreeze last block + weighted CE loss
python Task2/train_approach_b.py --backbone efficientnet_b0 --unfreeze --weighted

# Full encoder fine-tune from SupCon init (very low encoder LR = 1e-6)
python Task2/train_approach_b.py --backbone efficientnet_b0 --unfreeze_all
```

Saves `checkpoints_b_<backbone>[_unfrozen][_unfrozen_all][_weighted]/best_model_b.pth`.

### Evaluation on test set

```bash
# Evaluate all available checkpoints
python Task2/evaluate_test.py

# With test-time augmentation (4-view: orig + hflip + vflip + both)
python Task2/evaluate_test.py --tta
```

Saves confusion matrices and prints per-class classification reports to `test_results/`.

## Results Summary

| Method | Val Acc | Test Acc | Silhouette |
|---|---|---|---|
| Baseline | 0.7083 | — | — |
| **Approach A** (EfficientNet-B0, weighted CE) | **0.7514** | **0.7411** | — |
| Approach B — ResNet-18 frozen | 0.6471 | 0.6427 | 0.0293 |
| Approach B — ResNet-50 frozen | 0.6562 | 0.6510 | 0.0441 |
| Approach B — EfficientNet-B0 frozen | 0.6642 | 0.6527 | 0.0519 |
| Approach B — EfficientNet-B0 unfrozen | 0.6755 | 0.6684 | 0.0612 |
| Approach B — EfficientNet-B0 unfrozen + weighted | 0.6857 | 0.6561 | −0.0156 |

## Notes

- Test set (`Task2_Test_Set/`) is never touched during training or validation.
- Contrastive pre-training uses a separate subset of the training folder with no overlap with the 2500-per-class train split (no data leakage).
- Patch size 100×100 matches the test set format and covers the 99th-percentile nucleus diameter (~55 px) with ample context.
- Approach A beats the 0.7083 baseline by +3.3 pp on the test set.
