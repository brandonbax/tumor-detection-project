# Task 2 - Nuclei Classification

Classifies H&E nuclei patches into 3 classes: histiocyte, lymphocyte, tumor.

## Setup

Get a GPU node and activate your environment:

```bash
srun -p Teaching --nodelist=saxa --gres=gpu:1g.18gb:1 --cpus-per-task=2 --mem=128G --pty bash
conda activate cv
pip install -r Task2/requirements.txt
```

## How to run

Run everything from the repo root. Do step 0 first, everything else depends on it.

**Step 0 - Prepare the dataset (run this first)**
```bash
python Task2/dataset_preparation.py
```
Extracts 100x100 patches from the GeoJSON annotations and saves them as .npy files.

---

**Approach A - End-to-end classifier**
```bash
python Task2/train_approach_a.py
```
Fine-tunes EfficientNet-B0 with weighted cross-entropy. Saves checkpoint to `Task2/checkpoints_a/`.

---

**Approach B - SupCon pre-training then linear probe**

Step 1: pre-train the encoder
```bash
python Task2/pretrain_supcon.py --backbone efficientnet_b0
```

Step 2: train the classifier head (pick one)
```bash
# frozen encoder
python Task2/train_approach_b.py --backbone efficientnet_b0

# unfreeze last block
python Task2/train_approach_b.py --backbone efficientnet_b0 --unfreeze

# unfreeze everything
python Task2/train_approach_b.py --backbone efficientnet_b0 --unfreeze_all
```

---

**Evaluate on test set**
```bash
python Task2/evaluate_test.py
```
Saves results to `Task2/test_results/results_summary.json` and confusion matrix plots.
