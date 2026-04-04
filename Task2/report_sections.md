# Task 2 — Report Notes & Findings

## Dataset (Task 2.1)

- **Source:** PUMA dataset, train/validation splits
- **Classes (3 only, as per spec):** nuclei_histiocyte, nuclei_lymphocyte, nuclei_tumor
- **Patch size:** 100×100 px RGB, centred on nucleus centroid, zero-padded at borders
  - Justified by test set format (pre-made 100×100 patches) and exploration showing 99th percentile nucleus size ≈ 55px
- **Train set:** 2500 patches/class → 7500 total (perfectly balanced)
- **Validation set:** 700 patches/class → 2100 total (perfectly balanced)
- **Contrastive set:** 59,763 patches (remaining after train sampling — histiocyte: 3323, lymphocyte: 15150, tumor: 41290)
  - Imbalanced by design — reflects natural class frequency; labels not used during contrastive pre-training
  - Zero overlap with train set guaranteed by seeded shuffle (seed=42, deterministic)

### Original class imbalance (available patches)
| Class | Train available | Val available |
|---|---|---|
| nuclei_tumor | 43,790 | 5,800 |
| nuclei_lymphocyte | 17,650 | 2,351 |
| nuclei_histiocyte | 5,823 | 887 |

Histiocyte is the limiting class — caps the balanced train set at 2500/class.

---

## Approach A: End-to-End ResNet-18 Classifier

### Architecture
- ResNet-18 pretrained on ImageNet (transfer learning)
- Final FC replaced: 512 → 3 classes
- All layers fine-tuned (not frozen)
- ~11.2M trainable parameters

### Training setup
- Loss: CrossEntropy
- Optimiser: Adam (lr=1e-4)
- Scheduler: ReduceLROnPlateau (patience=4, factor=0.5)
- Early stopping: patience=7 on val_acc
- Batch size: 64, max 50 epochs

### Augmentations
- Random horizontal + vertical flip
- Random rotation ±15°
- ColorJitter (brightness=0.2, contrast=0.2, saturation=0.1)
- Normalised with ImageNet mean/std
- Rationale: H&E staining varies across slides; colour jitter improves robustness

### Results

| Run | Val Accuracy | Notes |
|---|---|---|
| Run 1 (local, no early stopping) | **0.7548** | 30 epochs, stopped at epoch 30 |
| Run 2 (cluster, early stop on val_loss) | 0.7295 | Stopped epoch 11 — wrong metric |
| Run 3 (cluster, early stop on val_acc) | **0.7433** | Stopped epoch 12 |

**Best result: 0.7548** (Run 1) — baseline is 0.7083, beats by **+4.7%**

### Per-class breakdown (best run)
| Class | Precision | Recall | F1 |
|---|---|---|---|
| nuclei_histiocyte | 0.7159 | 0.7129 | 0.7144 |
| nuclei_lymphocyte | 0.7364 | 0.8100 | 0.7714 |
| nuclei_tumor | 0.8199 | 0.7414 | 0.7787 |
| **Overall accuracy** | | | **0.7548** |

### Key observations
- Histiocyte consistently weakest — rarest class, visually similar to others
- Train acc reached ~0.89–0.90 while val acc plateaued ~0.74 → moderate overfitting
- Overfitting gap suggests the 7500-sample train set is small relative to ResNet-18's capacity
- Tumor has highest precision (0.82) — model is confident when predicting tumor
- Lymphocyte has best recall (0.81) — model rarely misses lymphocytes

---

## Approach B: SimCLR → Frozen Encoder + Linear Head

### Architecture
- Encoder: ResNet-18 (ImageNet pretrained) — projection head removed after pre-training
- Projection head (pre-training only): MLP 512 → 256 → 128, L2-normalised output
- Classifier head (fine-tuning): Linear 512 → 3, encoder frozen

### SimCLR Pre-training
- Dataset: 59,763 contrastive patches (no labels used)
- Loss: NT-Xent (Normalised Temperature-scaled Cross Entropy), τ=0.5
- Two randomly augmented views per image; positive pair = same image
- Augmentations: RandomResizedCrop(96), HorizontalFlip, ColorJitter(p=0.8), Grayscale(p=0.2), GaussianBlur(p=0.5)
- Optimiser: Adam (lr=3e-4), CosineAnnealingLR scheduler
- Batch size: 256, max 100 epochs, early stopping patience=10 on loss

### Linear Head Training
- Only classifier layer trained (512 → 3), encoder fully frozen
- Loss: CrossEntropy, Adam lr=1e-3 (higher LR appropriate for single linear layer)
- Early stopping patience=7 on val_acc

### Results
*(To be updated after Approach B completes)*

### Latent Space Evaluation
*(To be updated — t-SNE plot + silhouette score)*

---

## Comparison & Discussion (to complete)

| | Approach A | Approach B |
|---|---|---|
| Trainable params (training) | ~11.2M | ~1.5K (head only) |
| Val accuracy | 0.7548 | TBD |
| Beats baseline (0.7083)? | ✓ | TBD |
| Overfitting? | Yes (train 0.90 vs val 0.75) | Expected less — frozen encoder |

### Points to discuss in report
- Pre-training hypothesis: contrastive learning on unlabelled data should learn class-discriminative features, reducing overfitting when labelled data is scarce
- Histiocyte performance — imbalanced in contrastive set (5.6%) vs tumor (69.1%); does this affect latent space quality?
- Primary vs metastatic origin: dataset contains both; worth checking if misclassifications cluster by slide origin
- Why transfer learning from ImageNet works despite domain gap (natural images vs H&E stained histology)
- Parameter count: Approach A ~11.2M > baseline ~5M; Approach B head only ~1.5K

---

## Figures needed for report
- [ ] Training curves — loss & accuracy vs epoch (Approach A) → `checkpoints_a/training_curves_a.png`
- [ ] Training curves — loss & accuracy vs epoch (Approach B) → `checkpoints_b/training_curves_b.png`
- [ ] SimCLR pre-training loss curve → add to pretrain_simclr.py
- [ ] t-SNE of encoder features (Approach B) → `checkpoints_b/tsne_simclr.png`
- [ ] Example patches per class (from dataset_exploration.ipynb)
- [ ] Confusion matrix for both approaches
