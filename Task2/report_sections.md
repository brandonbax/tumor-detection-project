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

## Approach A: End-to-End EfficientNet-B0 Classifier

### Architecture
- EfficientNet-B0 pretrained on ImageNet (transfer learning)
- Final classifier replaced: Linear(1280, 3)
- All layers fine-tuned (not frozen)
- ~5.3M trainable parameters (closest match to ~5M baseline)

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
- Overfitting gap suggests the 7500-sample train set is small relative to EfficientNet-B0's capacity
- Tumor has highest precision (0.82) — model is confident when predicting tumor
- Lymphocyte has best recall (0.81) — model rarely misses lymphocytes

---

## Approach B: SupCon → Frozen Encoder + Linear Head

### Architecture (final: EfficientNet-B0)
- Encoder: EfficientNet-B0 (ImageNet pretrained) — projection head removed after pre-training
- Projection head (pre-training only): MLP 1280 → 256 → 128, L2-normalised output
- Classifier head (fine-tuning): Linear 1280 → 3, encoder fully frozen

### SupCon Pre-training
- Dataset: 59,763 contrastive patches (labels used to define positives)
- Loss: Supervised Contrastive Loss (Khosla et al., 2020), τ=0.07
- Two randomly augmented views per image; positives = all same-class patches in batch
- Augmentations: RandomResizedCrop(96), HorizontalFlip, ColorJitter(p=0.8), Grayscale(p=0.2), GaussianBlur(p=0.5)
- Optimiser: Adam (lr=3e-4), CosineAnnealingLR scheduler
- Batch size: 256, max 100 epochs, early stopping patience=10 on loss

### Linear Head Training
- Only classifier layer trained (1280 → 3), encoder fully frozen
- Loss: CrossEntropy, Adam lr=1e-3 (higher LR appropriate for single linear layer)
- Early stopping patience=7 on val_acc

## Approach B — Backbone Comparison

### ResNet-18 + SupCon (baseline backbone)
| Metric | Val | Test |
|---|---|---|
| Overall accuracy | 0.6443 | 0.6416 |
| Silhouette score | -0.0270 | — |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| nuclei_histiocyte | 0.4659 | 0.3581 | 0.4049 |
| nuclei_lymphocyte | 0.7032 | 0.6600 | 0.6809 |
| nuclei_tumor | 0.6667 | 0.8086 | 0.7308 |

### ResNet-50 + SupCon
| Metric | Val | Test |
|---|---|---|
| Overall accuracy | 0.6510 | TBD |
| Silhouette score | -0.0348 | — |
| Early stopping at epoch | 12 | — |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| nuclei_histiocyte | 0.6851 | 0.5314 | 0.5986 |
| nuclei_lymphocyte | 0.6008 | 0.6771 | 0.6367 |
| nuclei_tumor | 0.6784 | 0.7443 | 0.7098 |

### EfficientNet-B0 + SupCon (frozen encoder)
| Metric | Val | Test |
|---|---|---|
| Overall accuracy | 0.6714 | TBD |
| Silhouette score | -0.0190 | — |
| Early stopping at epoch | 15 | — |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| nuclei_histiocyte | 0.6857 | 0.5143 | 0.5878 |
| nuclei_lymphocyte | 0.6104 | 0.7229 | 0.6619 |
| nuclei_tumor | 0.7292 | 0.7771 | 0.7524 |

### EfficientNet-B0 + SupCon (unfrozen last block) ✓ Best Approach B
| Metric | Val | Test |
|---|---|---|
| Overall accuracy | **0.6890** | TBD |
| Silhouette score | -0.0141 | — |
| Early stopping at epoch | 21 | — |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| nuclei_histiocyte | 0.7187 | 0.5329 | 0.6120 |
| nuclei_lymphocyte | 0.6394 | 0.7700 | 0.6986 |
| nuclei_tumor | 0.7249 | 0.7643 | 0.7441 |

- Unfreezing `features[7]` + `features[8]` (last MBConv block + head conv) with differential LR (encoder: 1e-5, head: 1e-3)
- +1.76% over frozen variant; silhouette improves from -0.0190 → -0.0141

### Backbone + freeze ablation summary
| Backbone | Freeze | Val Acc | Test Acc | Silhouette |
|---|---|---|---|---|
| ResNet-18 | frozen | 0.6443 | 0.6416 | -0.0270 |
| ResNet-50 | frozen | 0.6510 | 0.6351 | -0.0348 |
| EfficientNet-B0 | frozen | 0.6714 | 0.6642 | -0.0190 |
| **EfficientNet-B0** | **last block unfrozen** | **0.6890** | **0.6755** | **-0.0141** |
| EfficientNet-B0 | unfrozen + weighted | 0.6857 | 0.6561 | -0.0156 |

**Winner: EfficientNet-B0 with last block unfrozen** — best val and test accuracy, best silhouette.

**Weighted loss hurt Approach B** (test 0.6755 → 0.6561): boosted histiocyte recall (0.53→0.70) but collapsed lymphocyte recall (0.67→0.52) — the contrastive encoder represents histiocyte and lymphocyte in overlapping space, so pushing histiocyte up pulls lymphocyte predictions down. Unlike Approach A where the full encoder can compensate, the partially frozen encoder cannot rebalance.

---

### SimCLR Results (original, replaced)
| Metric | Value |
|---|---|
| Overall val accuracy | 0.5995 |
| Silhouette score | -0.0139 |

### SupCon Results (final)
| Metric | Value |
|---|---|
| Overall val accuracy | **0.6443** |
| vs SimCLR | +4.5% improvement |
| vs baseline (0.7083) | below baseline |
| vs Approach A (0.7514) | worse by 10.7% |
| Early stopping at epoch | 21 |

#### Per-class breakdown (val set)
| Class | Precision | Recall | F1 |
|---|---|---|---|
| nuclei_histiocyte | 0.6761 | 0.4771 | 0.5595 |
| nuclei_lymphocyte | 0.6038 | 0.6900 | 0.6440 |
| nuclei_tumor | 0.6650 | 0.7657 | 0.7118 |
| **Overall** | 0.6483 | **0.6443** | 0.6384 |

### Latent Space Evaluation (SupCon)
- **Silhouette score: -0.0270** — still near 0, classes still overlapping in encoder feature space
- SupCon improved accuracy over SimCLR (+4.5%) but silhouette paradoxically worse
- Key insight: silhouette measures linear separability in 512-d feature space — near-0 score means the frozen linear head is the bottleneck, not the encoder quality per se
- t-SNE plot saved: `checkpoints_b/tsne_simclr.png`

---

## Contrastive Method Choice: SimCLR → SupCon

### Initial choice: SimCLR
SimCLR (Chen et al., 2020) was chosen initially:
- **Simplicity:** No memory bank or momentum encoder required; trains end-to-end in one stage
- **Large unlabelled set:** SimCLR benefits from large batches — our 59k contrastive set is well-suited
- **Well understood:** Extensive literature makes design choices interpretable

**SimCLR failed** — val=0.5995 (below random chance equivalent), silhouette=-0.0139. Cause: nuclei patches are visually very similar across classes; SimCLR's augmentation-based positives give the encoder no signal to distinguish them.

### Why SupCon instead
SupCon (Khosla et al., 2020) uses class labels to define positives — all same-class patches are pulled together in feature space. This directly addresses SimCLR's failure mode:
- Encoder is explicitly trained to separate histiocyte, lymphocyte, and tumor representations
- More positives per anchor (N_class × 2 views) vs SimCLR (just 1 positive per anchor)
- Temperature τ=0.07 (vs 0.5 for SimCLR) creates sharper distributions and stronger gradients

### Alternatives considered
| Method | Key idea | Why not chosen |
|---|---|---|
| **MoCo v2** | Momentum encoder + memory bank for large effective batch | More complex; memory bank implementation overhead |
| **BYOL** | No negative pairs; bootstrap target network | More complex (stop-gradient, EMA); no label supervision |
| **DINO** | Self-distillation with Vision Transformers | ViT overkill for 100×100 patches; heavy compute |

### Relation to course tutorials
The course labs covered classical feature-based methods (Lab 3: SIFT + Bag of Words + SVM, Lab 4: HOG + SVM). Our approaches extend this:
- **Approach A** replaces hand-crafted SIFT/HOG features with learned CNN features (ResNet-18) — same classification paradigm, better features
- **Approach B** goes further: self-supervised pre-training learns features from unlabelled data, then a linear SVM-style head classifies — conceptually similar to Lab 3's BoW pipeline but with deep representations

---

## Key Discussion Points (for report)

### Convergence behaviour
- Early stopping consistently triggers at epoch 11–12 across all runs — the model genuinely saturates fast on 7500 samples
- This is expected: EfficientNet-B0 has ~5.3M parameters trained on only 7500 samples — the model has capacity to overfit quickly
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

### Approach A — EfficientNet-B0 + Weight=3.0 (latest run, val set)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| nuclei_histiocyte | 0.7171 | 0.7171 | 0.7171 | 700 |
| nuclei_lymphocyte | 0.7375 | 0.7986 | 0.7668 | 700 |
| nuclei_tumor | 0.8053 | 0.7386 | 0.7705 | 700 |
| **Overall** | 0.7533 | **0.7514** | 0.7515 | 2100 |

- Early stopping at epoch 24 (train_acc=0.91 vs val_acc=0.75 — overfitting still present)
- Histiocyte significantly improved: recall 0.49 → **0.72** vs original ResNet-18 run

---

### Validation Set Results
| | Baseline | Approach A (EfficientNet-B0) | Approach B best (EfficientNet-B0 unfrozen) |
|---|---|---|---|
| Trainable params | ~5M | ~5.3M | ~1.5M (last block + head) |
| Val accuracy | 0.7083 | **0.7514** | 0.6890 |
| Test accuracy | 0.7083 | **0.7411** | 0.6755 |
| Beats baseline? | — | ✓ (+3.3% test) | ✗ (-3.3% test) |
| Silhouette score | — | N/A | -0.0141 |

### Test Set Results — Final (Task2_Test_Set, 1858 patches: 458 histiocyte / 700 lymphocyte / 700 tumor)
| | Baseline | Approach A (EfficientNet-B0) | Approach B best (EfficientNet-B0 unfrozen) |
|---|---|---|---|
| Overall accuracy | 0.7083 | **0.7411** | 0.6755 |
| Beats baseline? | — | ✓ (+3.3%) | ✗ (-3.3%) |

#### Approach A — Per-class (test set)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| nuclei_histiocyte | 0.5916 | 0.7052 | 0.6434 | 458 |
| nuclei_lymphocyte | 0.8257 | 0.6700 | 0.7397 | 700 |
| nuclei_tumor | 0.7863 | 0.8357 | 0.8102 | 700 |
| **Overall** | 0.7531 | **0.7411** | 0.7426 | 1858 |

#### Approach B best (EfficientNet-B0 unfrozen) — Per-class (test set)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| nuclei_histiocyte | 0.5139 | 0.5262 | 0.5200 | 458 |
| nuclei_lymphocyte | 0.7386 | 0.6700 | 0.7026 | 700 |
| nuclei_tumor | 0.7228 | 0.7786 | 0.7497 | 700 |
| **Overall** | 0.6772 | **0.6755** | 0.6753 | 1858 |

### Val vs Test gap
- Val balanced (700/700/700); test imbalanced (458/700/700)
- Histiocyte fewest samples and hardest class — drives val/test gap
- Weighted loss (histiocyte=3.0) helped close the gap significantly

### Points to discuss in report
- Pre-training hypothesis: contrastive learning on unlabelled data should learn class-discriminative features, reducing overfitting when labelled data is scarce — **this hypothesis was NOT supported by results**
- SimCLR failed because: (1) nuclei patches are visually very similar across classes, (2) contrastive set is 69% tumor — encoder biased towards tumor features, (3) SimCLR's aggressive augmentations may destroy subtle discriminative morphological features
- Silhouette score ≈ 0 confirms encoder learned no class-separable structure — the frozen encoder is fundamentally limited
- Histiocyte worst recall (0.37) — rarest in contrastive set (5.6%), encoder never learned to represent it well
- Approach B below baseline is a valid and interesting result — it shows self-supervised contrastive learning is not always better, especially on domain-specific data with subtle inter-class differences
- Primary vs metastatic origin: dataset contains both; worth checking if misclassifications cluster by slide origin
- Why transfer learning from ImageNet works despite domain gap (natural images vs H&E stained histology)
- Parameter count: Approach A ~5.3M ≈ baseline ~5M; Approach B head only ~3.84K (1280×3 linear, encoder frozen)

---

---

## References (BibTeX)

```bibtex
% SimCLR
@inproceedings{chen2020simclr,
  author    = {Ting Chen and Simon Kornblith and Mohammad Norouzi and Geoffrey Hinton},
  title     = {A Simple Framework for Contrastive Learning of Visual Representations},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {119},
  pages     = {1597--1607},
  publisher = {PMLR},
  year      = {2020},
  eprint    = {2002.05709},
  archivePrefix = {arXiv}
}

% SupCon
@inproceedings{khosla2020supcon,
  author    = {Prannay Khosla and Piotr Teterwak and Chen Wang and Aaron Sarna and
               Yonglong Tian and Phillip Isola and Aaron Maschinot and Ce Liu and
               Dilip Krishnan},
  title     = {Supervised Contrastive Learning},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  pages     = {18661--18673},
  year      = {2020},
  eprint    = {2004.11362},
  archivePrefix = {arXiv}
}

% ResNet
@inproceedings{he2016resnet,
  author    = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {770--778},
  year      = {2016},
  doi       = {10.1109/CVPR.2016.90},
  eprint    = {1512.03385},
  archivePrefix = {arXiv}
}

% EfficientNet
@inproceedings{tan2019efficientnet,
  author    = {Mingxing Tan and Quoc V. Le},
  title     = {{EfficientNet}: Rethinking Model Scaling for Convolutional Neural Networks},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {97},
  pages     = {6105--6114},
  publisher = {PMLR},
  year      = {2019},
  eprint    = {1905.11946},
  archivePrefix = {arXiv}
}

% HoVer-Net — nuclei segmentation & classification in histopathology
@article{graham2019hovernet,
  author    = {Simon Graham and Quoc Dang Vu and Shan E Ahmed Raza and Ayesha Azam and
               Yee Wah Tsang and Jin Tae Kwak and Nasir Rajpoot},
  title     = {{HoVer-Net}: Simultaneous Segmentation and Classification of Nuclei in
               Multi-Tissue Histology Images},
  journal   = {Medical Image Analysis},
  volume    = {58},
  pages     = {101563},
  year      = {2019},
  doi       = {10.1016/j.media.2019.101563},
  eprint    = {1812.06499},
  archivePrefix = {arXiv}
}

% Contrastive SSL Survey
@article{jaiswal2021survey,
  author    = {Ashish Jaiswal and Ashwin Ramesh Babu and Mohammad Zaki Zadeh and
               Debapriya Banerjee and Fillia Makedon},
  title     = {A Survey on Contrastive Self-Supervised Learning},
  journal   = {Technologies},
  volume    = {9},
  number    = {1},
  pages     = {2},
  year      = {2021},
  doi       = {10.3390/technologies9010002},
  eprint    = {2011.00362},
  archivePrefix = {arXiv}
}

% Transfer learning for medical imaging
@article{tajbakhsh2016transfer,
  author    = {Nima Tajbakhsh and Jae Y. Shin and Suryakanth R. Gurudu and
               R. Todd Hurst and Christopher B. Kendall and Michael B. Gotway and
               Jianming Liang},
  title     = {Convolutional Neural Networks for Medical Image Analysis:
               Full Training or Fine Tuning?},
  journal   = {IEEE Transactions on Medical Imaging},
  volume    = {35},
  number    = {5},
  pages     = {1299--1312},
  year      = {2016},
  doi       = {10.1109/TMI.2016.2535302}
}

% Deep learning for computational histopathology survey
@article{srinidhi2021histopathology,
  author    = {Chetan L. Srinidhi and Ozan Ciga and Anne L. Martel},
  title     = {Deep Neural Network Models for Computational Histopathology: A Survey},
  journal   = {Medical Image Analysis},
  volume    = {67},
  pages     = {101813},
  year      = {2021},
  doi       = {10.1016/j.media.2020.101813},
  eprint    = {1912.12378},
  archivePrefix = {arXiv}
}

% Self-supervised contrastive learning for digital histopathology
@article{ciga2022selfsuphisto,
  author    = {Ozan Ciga and Tony Xu and Anne Louise Martel},
  title     = {Self Supervised Contrastive Learning for Digital Histopathology},
  journal   = {Machine Learning with Applications},
  volume    = {7},
  pages     = {100198},
  year      = {2022},
  doi       = {10.1016/j.mlwa.2021.100198},
  eprint    = {2011.13971},
  archivePrefix = {arXiv}
}

% t-SNE
@article{vandermaaten2008tsne,
  author    = {Laurens van der Maaten and Geoffrey Hinton},
  title     = {Visualizing Data using {t-SNE}},
  journal   = {Journal of Machine Learning Research},
  volume    = {9},
  pages     = {2579--2605},
  year      = {2008},
  url       = {https://www.jmlr.org/papers/v9/vandermaaten08a.html}
}

% UMAP
@article{mcinnes2018umap,
  author    = {Leland McInnes and John Healy and James Melville},
  title     = {{UMAP}: Uniform Manifold Approximation and Projection for Dimension Reduction},
  journal   = {arXiv preprint},
  year      = {2018},
  eprint    = {1802.03426},
  archivePrefix = {arXiv}
}
```

---

## Figures needed for report
- [ ] Training curves — loss & accuracy vs epoch (Approach A) → `checkpoints_a/training_curves_a.png`
- [ ] Training curves — loss & accuracy vs epoch (Approach B, EfficientNet-B0) → `checkpoints_b_efficientnet_b0/training_curves_b_efficientnet_b0.png`
- [ ] SupCon pre-training loss curve → `checkpoints_b_efficientnet_b0/supcon_pretrain_loss.png`
- [ ] t-SNE of encoder features (Approach B, EfficientNet-B0) → `checkpoints_b_efficientnet_b0/tsne_efficientnet_b0.png`
- [ ] Example patches per class (from dataset_exploration.ipynb)
- [ ] Confusion matrix — Approach A → `test_results/confusion_matrix_a.png`
- [ ] Confusion matrix — Approach B → `test_results/confusion_matrix_b_efficientnet_b0.png`
