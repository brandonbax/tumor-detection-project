"""
config.py - Central configuration for Task 2.
All hyperparameters and paths live here so training scripts stay short.
"""
from pathlib import Path
import torch

# Paths
ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "task2_dataset"
TEST_DIR   = ROOT_DIR.parent / "Task2_Test_Set"
CKPT_A_DIR = ROOT_DIR / "checkpoints_a"

# Classes (Task 2 spec: exactly these 3)
TARGET_CLASSES = ["nuclei_histiocyte", "nuclei_lymphocyte", "nuclei_tumor"]
NUM_CLASSES    = 3

# Approach A
A_BATCH_SIZE     = 64
A_LR             = 1e-4
A_EPOCHS         = 50
A_EARLY_STOP_PAT = 7
A_CLASS_WEIGHTS  = [3.0, 1.0, 1.0]  # histiocyte upweighted, rarest and hardest class

# SupCon pre-training (Approach B step 1)
SC_BATCH_SIZE     = 256
SC_LR             = 3e-4
SC_EPOCHS         = 100
SC_TEMPERATURE    = 0.07
SC_PROJ_DIM       = 128
SC_EARLY_STOP_PAT = 10

# Approach B fine-tuning (step 2)
B_BATCH_SIZE     = 64
B_LR             = 1e-3
B_EPOCHS         = 50
B_EARLY_STOP_PAT = 7

# Shared
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()          else
    torch.device("cpu")
)
