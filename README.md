# Medical Image Segmentation and Nuclei Classification

Deep-learning computer vision project for analysing H&E-stained pathology images from the Puma dataset. The project covers two connected problems: pixel-wise tissue segmentation and classification of individual nuclei. It compares standard supervised training with unsupervised or self-supervised representation learning, and includes end-to-end training, evaluation, visualisations, and saved model checkpoints.

## Why this project is relevant

This project demonstrates a practical machine-learning workflow rather than only a model definition:

- converting GeoJSON pathology annotations into raster segmentation masks and labelled image patches;
- designing and training custom PyTorch architectures for dense prediction;
- handling class imbalance with weighted losses and Dice-aware objectives;
- using augmentation, transfer learning, early stopping, learning-rate schedules, and reproducible seeds;
- evaluating models with task-appropriate metrics, confusion matrices, prediction grids, training curves, and t-SNE embeddings;
- comparing parameter counts and empirical trade-offs between model architectures.

## Project overview

### Task 1: tissue segmentation

The input is an RGB `.tif` pathology image and the output is a three-class mask:

1. `Tumor`
2. `Stroma`
3. `Other` - blood vessel, epidermis, white background, and necrosis

The GeoJSON annotations are rasterised into masks. Training uses 256 x 256 patches with joint image/mask augmentation, including flips, rotations, affine and elastic transforms, colour jitter, blur, and noise.

Two segmentation pipelines are implemented:

- **Attention ASPP U-Net** - a custom residual U-Net with attention-gated skip connections, an Atrous Spatial Pyramid Pooling bottleneck for multi-scale context, and deep-supervision outputs. It is trained end-to-end with a weighted combination of cross-entropy and Dice loss.
- **Autoencoder pre-training + segmentation decoder** - an autoencoder first learns to reconstruct raw images without labels. Its encoder is then frozen and connected to a segmentation decoder, testing whether reconstruction features transfer to tissue segmentation.

### Task 2: nuclei classification

The input is a 100 x 100 RGB patch centred on a nucleus. The classifier predicts one of:

- `Histiocyte`
- `Lymphocyte`
- `Tumor`

The dataset preparation script extracts 2,500 training patches and 700 validation patches per class from the GeoJSON nuclei annotations. It also creates a separate contrastive-learning set from unused training patches to avoid data leakage.

Two classification approaches are implemented:

- **Approach A: supervised EfficientNet-B0** - ImageNet-pretrained EfficientNet-B0 is fine-tuned end-to-end with weighted cross-entropy, giving extra emphasis to the rare and difficult histiocyte class.
- **Approach B: supervised contrastive pre-training + classifier** - an encoder is pre-trained with supervised contrastive loss, then used with a linear classification head. The repository supports a frozen encoder, partial unfreezing, and full fine-tuning with differential learning rates. EfficientNet-B0 and ResNet-50 backbones are available.

## Results

The saved test-set results show the behaviour of each design, including cases where a more sophisticated pre-training strategy did not outperform direct supervised fine-tuning.

### Tissue segmentation

| Model | Pixel accuracy | Mean Dice | Mean IoU | Trainable parameters |
| --- | ---: | ---: | ---: | ---: |
| Attention ASPP U-Net | 0.7952 | **0.5846** | **0.4894** | 69.3M |
| Frozen autoencoder encoder + decoder | 0.6390 | 0.3909 | 0.3131 | 45.8M decoder |

The U-Net exceeds the assignment baseline mean Dice of 0.4670. Its per-class Dice scores were 0.8355 for tumor, 0.6168 for stroma, and 0.3014 for `Other`, illustrating the impact of class imbalance and the heterogeneity of the merged `Other` category.

### Nuclei classification

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| EfficientNet-B0, supervised | **0.7126** | **0.7035** | **0.7042** | **0.7008** |
| EfficientNet-B0, contrastive encoder, frozen | 0.6792 | 0.6526 | 0.6508 | 0.6499 |
| EfficientNet-B0, contrastive encoder, partially unfrozen | 0.6948 | 0.6704 | 0.6666 | 0.6655 |
| EfficientNet-B0, contrastive encoder, fully unfrozen | 0.6808 | 0.6575 | 0.6557 | 0.6545 |
| ResNet-50, contrastive encoder, frozen | 0.6351 | 0.6090 | 0.6057 | 0.6044 |

The supervised EfficientNet-B0 is above the assignment baseline accuracy of 0.7083. Tumor nuclei were the strongest class, while histiocytes remained the most challenging, motivating weighted loss experiments and class-level precision/recall analysis.

## Repository structure

```text
.
├── Dataset_Splits/             # Train/validation pathology images and annotations
├── Task2/                      # Nuclei patch extraction, classifiers, SupCon, evaluation
│   ├── dataset_preparation.py
│   ├── train_approach_a.py
│   ├── pretrain_supcon.py
│   ├── train_approach_b.py
│   ├── evaluate_test.py
│   ├── results/
│   └── test_results/
├── Task2_Test_Set/             # Held-out nuclei patches
├── src/task1/                  # Tissue segmentation package
│   ├── data_processing.py
│   ├── augmentation.py
│   ├── models.py
│   ├── train_unet.py
│   ├── train_autoencoder.py
│   ├── train_ae_seg.py
│   └── evaluate.py
├── src/task1/results/          # Metrics and generated evaluation artefacts
├── tests/                      # Automated tests
├── environment.yml             # Conda environment for Task 1
└── run_task1.sh                # Full Task 1 pipeline wrapper
```

## Reproducing the experiments

### Environment

For the segmentation pipeline:

```bash
conda env create -f environment.yml
conda activate cv-cw
```

For the nuclei-classification pipeline:

```bash
conda create -n cv python=3.10 -y
conda activate cv
pip install -r Task2/requirements.txt
```

Training is intended to run with CUDA when available. `check_cuda.py` performs a PyTorch/CUDA availability check and a GPU matrix-multiplication smoke test.

### Run Task 1

Run the complete segmentation workflow, including training both models and evaluating them on the held-out test set:

```bash
bash run_task1.sh --use_class_weights --dampen_weights
```

The wrapper runs:

1. end-to-end U-Net training;
2. image autoencoder pre-training;
3. frozen-encoder AE-Seg training;
4. test-set evaluation and comparison plots.

To evaluate existing final checkpoints after cloning:

```bash
git lfs install
git lfs pull
python -m task1.evaluate \
  --unet_ckpt src/task1/checkpoints/unet_final.pth \
  --ae_seg_ckpt src/task1/checkpoints/ae_seg_final.pth
```

### Run Task 2

Prepare the 100 x 100 nucleus patches:

```bash
python Task2/dataset_preparation.py
```

Train the supervised baseline:

```bash
python Task2/train_approach_a.py
```

Train the contrastive-learning pipeline:

```bash
python Task2/pretrain_supcon.py --backbone efficientnet_b0
python Task2/train_approach_b.py --backbone efficientnet_b0
```

Optional fine-tuning variants are available:

```bash
python Task2/train_approach_b.py --backbone efficientnet_b0 --unfreeze
python Task2/train_approach_b.py --backbone efficientnet_b0 --unfreeze_all
```

Evaluate saved checkpoints on the provided test set:

```bash
python Task2/evaluate_test.py
```

Final checkpoints are tracked with Git LFS. After `git lfs pull`, evaluation can be run without retraining.

## Outputs

Evaluation scripts save quantitative and qualitative artefacts alongside the checkpoints:

- segmentation metrics, class distributions, confusion matrices, prediction grids, and model-comparison plots in `src/task1/results/`;
- classification training curves and confusion matrices in `Task2/results/`;
- classification metrics in `Task2/test_results/results_summary.json`;
- t-SNE plots for comparing the learned contrastive latent spaces.

## Technologies

Python, PyTorch, Torchvision, TorchMetrics, Albumentations, GeoPandas, Rasterio, tifffile, NumPy, scikit-learn, Matplotlib, and Seaborn.
