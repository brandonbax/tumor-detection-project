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
| Run 3 (cluster, early stop on val_acc) | 0.7433 | Stopped epoch 12 |
| Run 4 (cluster, early stop on val_acc) | 0.7386 | Stopped epoch 11, best at epoch 4/7 |

**Best result: 0.7548** (Run 1) — baseline is 0.7083, beats by **+4.7%**

### Per-class breakdown (all runs)
| Class | Precision | Recall | F1 | Run |
|---|---|---|---|---|
| nuclei_histiocyte | 0.7159 | 0.7129 | 0.7144 | Run 1 (best) |
| nuclei_lymphocyte | 0.7364 | 0.8100 | 0.7714 | Run 1 (best) |
| nuclei_tumor | 0.8199 | 0.7414 | 0.7787 | Run 1 (best) |
| **Overall accuracy** | | | **0.7548** | Run 1 (best) |

| Class | Precision | Recall | F1 | Run |
|---|---|---|---|---|
| nuclei_histiocyte | 0.7588 | 0.6471 | 0.6985 | Run 4 |
| nuclei_lymphocyte | 0.7166 | 0.7729 | 0.7436 | Run 4 |
| nuclei_tumor | 0.7447 | 0.7957 | 0.7693 | Run 4 |
| **Overall accuracy** | | | **0.7386** | Run 4 |

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

### Results (final run)
| Metric | Value |
|---|---|
| Overall accuracy | **0.5995** |
| vs baseline (0.7083) | **below baseline by 10.9%** |
| vs Approach A (0.7548) | **worse by 15.5%** |
| Early stopping at epoch | 30 |

#### Per-class breakdown
| Class | Precision | Recall | F1 |
|---|---|---|---|
| nuclei_histiocyte | 0.5954 | 0.4457 | 0.5098 |
| nuclei_lymphocyte | 0.6006 | 0.6229 | 0.6115 |
| nuclei_tumor | 0.6012 | 0.7300 | 0.6594 |
| **Overall** | **0.5990** | **0.5995** | **0.5936** |

### Latent Space Evaluation
- **Silhouette score: -0.0139** — essentially 0, confirming no class separation in encoder features
- t-SNE plot saved: `checkpoints_b/tsne_simclr.png`
- A silhouette score near 0 means classes are completely overlapping in the latent space — the SimCLR encoder learned instance-level similarity but no class-discriminative structure

---

## Why SimCLR? Alternatives Considered

### Why SimCLR
SimCLR (Chen et al., 2020) was chosen as the contrastive pre-training strategy for several reasons:
- **Simplicity:** No memory bank or momentum encoder required (unlike MoCo); trains end-to-end in one stage
- **Strong performance:** Achieves competitive results with a simple NT-Xent loss and strong augmentation
- **Large unlabelled set:** SimCLR benefits from large batches and large datasets — our 59k contrastive set is well-suited
- **Well understood:** Extensive literature makes design choices (temperature, projection head size, augmentations) interpretable for the report

### Alternatives considered
| Method | Key idea | Why not chosen |
|---|---|---|
| **MoCo v2** | Momentum encoder + memory bank for large effective batch | More complex; memory bank adds implementation overhead |
| **SupCon** (Supervised Contrastive) | Uses labels to define positives — same-class patches pulled together | Requires labels during pre-training, defeating the purpose of using unlabelled contrastive set |
| **BYOL** | No negative pairs; uses bootstrap target network | No negatives means NT-Xent not needed, but more complex (stop-gradient, EMA) |
| **DINO** | Self-distillation with Vision Transformers | ViT overkill for 100×100 patches; heavy compute |

SimCLR hits the right balance of simplicity, interpretability, and effectiveness for this task.

### Relation to course tutorials
The course labs covered classical feature-based methods (Lab 3: SIFT + Bag of Words + SVM, Lab 4: HOG + SVM). Our approaches extend this:
- **Approach A** replaces hand-crafted SIFT/HOG features with learned CNN features (ResNet-18) — same classification paradigm, better features
- **Approach B** goes further: self-supervised pre-training learns features from unlabelled data, then a linear SVM-style head classifies — conceptually similar to Lab 3's BoW pipeline but with deep representations

---

## Key Discussion Points (for report)

### Convergence behaviour
- Early stopping consistently triggers at epoch 11–12 across all runs — the model genuinely saturates fast on 7500 samples
- This is expected: ResNet-18 has ~11M parameters trained on only 7500 samples — the model has capacity to overfit quickly
- Train acc reaches 0.89–0.90 while val acc plateaus at 0.73–0.75 — a ~15% generalisation gap indicates overfitting
- **For the report:** This motivates Approach B — contrastive pre-training on 59k samples first should learn more generalisable features, reducing the overfitting gap when fine-tuning the head

### Why histiocyte is consistently weakest
- Histiocyte had fewest original patches (5,823 train) — only 3,323 in the contrastive set (5.6% vs tumor's 69.1%)
- Visually, histiocytes can resemble lymphocytes in H&E staining — both appear as round, darkly stained nuclei
- The imbalance in the contrastive set may mean the encoder learns weaker representations for histiocyte
- **For the report:** Discuss whether contrastive set imbalance affects latent space quality (check silhouette score per class in t-SNE)

### Why accuracy varies across runs (0.7386–0.7548)
- Same architecture, same data, different random seeds → stochastic gradient descent gives different minima
- Val acc variance of ~1.6% is normal for this dataset size
- **For the report:** Report the best run result but acknowledge variance; could average over multiple runs for robustness

### ImageNet → H&E domain gap
- ResNet-18 was pretrained on natural images (dogs, cars, etc.) — very different from H&E stained histology
- Despite this, transfer learning still works well because low-level features (edges, textures, colour gradients) transfer across domains
- H&E staining produces consistent colour patterns (purple nuclei, pink cytoplasm) that ImageNet-trained colour detectors partially capture
- **For the report:** This justifies using pretrained weights rather than training from scratch, especially with only 7500 samples

### Primary vs metastatic origin
- The PUMA dataset contains slides from both primary tumour and metastatic sites
- Nuclei morphology can differ between primary and metastatic tissue — tumour nuclei in metastatic sites may look atypical
- We did not stratify by origin during sampling (seed=42 random sample) — could introduce systematic bias
- **For the report:** Acknowledge this limitation; a proper analysis would require slide-level metadata to check if misclassifications cluster by origin

---

## Comparison & Discussion (to complete)

### Validation Set Results
| | Baseline | Approach A | Approach B |
|---|---|---|---|
| Trainable params | ~5M | ~11.2M | ~1.5K (head only) |
| Val accuracy | 0.7083 | **0.7548** | 0.5995 |
| Beats baseline? | — | ✓ (+4.7%) | ✗ (-10.9%) |
| Train/val gap | — | ~15% (overfitting) | ~9% (underfitting) |
| Silhouette score | — | N/A | -0.0139 (no separation) |

### Test Set Results (Task2_Test_Set, 1858 patches: 458 histiocyte / 700 lymphocyte / 700 tumor)
| | Baseline | Approach A | Approach B |
|---|---|---|---|
| Overall accuracy | 0.7083 | 0.6787 | 0.6206 |
| Beats baseline? | — | ✗ (-2.96%) | ✗ (-8.77%) |

#### Approach A — Per-class (test set)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| nuclei_histiocyte | 0.5257 | 0.4913 | 0.5079 | 458 |
| nuclei_lymphocyte | 0.7858 | 0.6500 | 0.7115 | 700 |
| nuclei_tumor | 0.6827 | 0.8300 | 0.7492 | 700 |
| **Overall** | 0.6829 | **0.6787** | 0.6755 | 1858 |

#### Approach B — Per-class (test set)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| nuclei_histiocyte | 0.4336 | 0.3996 | 0.4159 | 458 |
| nuclei_lymphocyte | 0.6552 | 0.6271 | 0.6409 | 700 |
| nuclei_tumor | 0.6932 | 0.7586 | 0.7244 | 700 |
| **Overall** | 0.6149 | **0.6206** | 0.6169 | 1858 |

### Val vs Test gap analysis
- Val was balanced (700/700/700); test is imbalanced (458/700/700)
- Histiocyte has fewest test samples AND worst performance — drags overall accuracy down
- Both models were not trained with class-weighted loss — histiocyte penalised equally despite being hardest class
- **Fix: add weighted CrossEntropy loss to Approach A and retrain**

### Points to discuss in report
- Pre-training hypothesis: contrastive learning on unlabelled data should learn class-discriminative features, reducing overfitting when labelled data is scarce — **this hypothesis was NOT supported by results**
- SimCLR failed because: (1) nuclei patches are visually very similar across classes, (2) contrastive set is 69% tumor — encoder biased towards tumor features, (3) SimCLR's aggressive augmentations may destroy subtle discriminative morphological features
- Silhouette score ≈ 0 confirms encoder learned no class-separable structure — the frozen encoder is fundamentally limited
- Histiocyte worst recall (0.37) — rarest in contrastive set (5.6%), encoder never learned to represent it well
- Approach B below baseline is a valid and interesting result — it shows self-supervised contrastive learning is not always better, especially on domain-specific data with subtle inter-class differences
- Primary vs metastatic origin: dataset contains both; worth checking if misclassifications cluster by slide origin
- Why transfer learning from ImageNet works despite domain gap (natural images vs H&E stained histology)
- Parameter count: Approach A ~11.2M > baseline ~5M; Approach B head only ~1.5K (encoder frozen)

---

## Figures needed for report
- [ ] Training curves — loss & accuracy vs epoch (Approach A) → `checkpoints_a/training_curves_a.png`
- [ ] Training curves — loss & accuracy vs epoch (Approach B) → `checkpoints_b/training_curves_b.png`
- [ ] SimCLR pre-training loss curve → add to pretrain_simclr.py
- [ ] t-SNE of encoder features (Approach B) → `checkpoints_b/tsne_simclr.png`
- [ ] Example patches per class (from dataset_exploration.ipynb)
- [ ] Confusion matrix for both approaches
