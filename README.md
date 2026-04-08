# computer-vision-coursework

All Task 2 code and scripts are in the `Task2/` folder. See [Task2/Readme.md](Task2/Readme.md) for full details.

---

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

All files are in the `Task2/` folder. See [Task2/Readme.md](Task2/Readme.md) for full details.

### Environment
Create and activate a conda environment:
```bash
conda create -n cv python=3.10 -y
conda activate cv
pip install -r Task2/requirements.txt
```

### Train models
```bash
python Task2/dataset_preparation.py
python Task2/train_approach_a.py
python Task2/pretrain_supcon.py --backbone efficientnet_b0
python Task2/train_approach_b.py --backbone efficientnet_b0 --unfreeze_all
```

### Testing on final weights
Weights are stored using git lfs:
```bash
git lfs install
git lfs pull
```
Then run:
```bash
python Task2/evaluate_test.py --tta
```
