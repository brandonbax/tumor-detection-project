import os

# ═══════════════════════════════════════════════
#  Data Paths
# ═══════════════════════════════════════════════
DATASET_ROOT = "Dataset_Splits"

TRAIN_IMAGE_DIR = os.path.join(DATASET_ROOT, "train", "image")
TRAIN_LABEL_DIR = os.path.join(DATASET_ROOT, "train", "tissue")

VAL_IMAGE_DIR = os.path.join(DATASET_ROOT, "validation", "image")
VAL_LABEL_DIR = os.path.join(DATASET_ROOT, "validation", "tissue")

TEST_IMAGE_DIR = os.path.join(DATASET_ROOT, "test", "image")
TEST_LABEL_DIR = os.path.join(DATASET_ROOT, "test", "tissue")

MASK_DIR = os.path.join("src", "task1", "masks")

# ═══════════════════════════════════════════════
#  Dataset Statistics (for normalization)
# ═══════════════════════════════════════════════
DATASET_MEAN = (0.6194304, 0.41257372, 0.69565123)
DATASET_STD = (0.17032179, 0.1604232, 0.11571649)

# ═══════════════════════════════════════════════
#  Classes
# ═══════════════════════════════════════════════
CLASS_NAME_TO_ID = {
    'tissue_tumor': 1,
    'tissue_stroma': 2,
}

# ═══════════════════════════════════════════════
#  Training Parameters
# ═══════════════════════════════════════════════
PATCH_SIZE = 256
NUM_WORKERS = 4

NUM_CLASSES = 3
CLASS_NAMES = ['Other', 'Tumor', 'Stroma']
IMAGE_CHANNELS = 3

SEED = 42
CHECKPOINT_DIR = os.path.join("src", "task1", "checkpoints")
RESULTS_DIR = os.path.join("src", "task1", "results")

UNET_EPOCHS = 50
UNET_BATCH_SIZE = 16
UNET_LR = 1e-4
UNET_WEIGHT_DECAY = 1e-4
UNET_FEATURES = [64, 128, 256, 512, 1024]

AE_EPOCHS = 30
AE_BATCH_SIZE = 32
AE_LR = 1e-3
AE_WEIGHT_DECAY = 1e-4
AE_FEATURES = [64, 128, 256, 512, 1024]

AE_SEG_EPOCHS = 50
AE_SEG_BATCH_SIZE = 8
AE_SEG_LR = 1e-4
AE_SEG_WEIGHT_DECAY = 1e-4

LR_MIN = 1e-6

# ═══════════════════════════════════════════════
#  Evaluation Baseline
# ═══════════════════════════════════════════════
BASELINE_DICE = 0.4670
BASELINE_PARAMS = 125_000_000
