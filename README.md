# computer-vision-coursework
## Task 1

### Environment
Setup environment and install dependencies by running:
```bash
./setup_env.sh
```
Assuming that conda is on PATH, activate the environment (if not already active) with:
```bash
conda activate cv-cw
```

### Train models
```bash
python -m task1.run_task1 --use_class_weights --dampen_weights
```

### Testing on final weights
Weights are stored using git lfs, so if not already done so, git lfs must be enabled to download the weights:
```bash
git lfs install
git lfs pull
```
Then run:
```bash
python -m task1.evaluate --unet_ckpt src/task1/checkpoints/unet_final.pth --ae_seg_ckpt src/task1/checkpoints/ae_seg_final.pth
```

---

## Task 2

For Task 2 (nuclei classification) setup, training scripts, evaluation, and model weights, see the dedicated readme:

👉 [Task2/Readme.md](Task2/Readme.md)

Quick start:
```bash
# Get a GPU node and set up environment
srun -p Teaching --nodelist=saxa --gres=gpu:1g.18gb:1 --cpus-per-task=2 --mem=128G --pty bash
conda activate cv
pip install -r Task2/requirements.txt

# Prepare dataset first
python Task2/dataset_preparation.py

# Then train or evaluate — see Task2/Readme.md for full details
```