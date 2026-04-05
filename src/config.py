import os

# ═══════════════════════════════════════════════
#  Data Paths
# ═══════════════════════════════════════════════
DATASET_ROOT = "Dataset_Splits"

TRAIN_IMAGE_DIR = os.path.join(DATASET_ROOT, "train", "image")
TRAIN_LABEL_DIR = os.path.join(DATASET_ROOT, "train", "label")

VAL_IMAGE_DIR = os.path.join(DATASET_ROOT, "val", "image")
VAL_LABEL_DIR = os.path.join(DATASET_ROOT, "val", "label")

TEST_IMAGE_DIR = os.path.join(DATASET_ROOT, "test", "image")
TEST_LABEL_DIR = os.path.join(DATASET_ROOT, "test", "label")

MASK_DIR = "masks"

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
UNET_BATCH_SIZE = 16
AE_BATCH_SIZE = 32
NUM_WORKERS = 4
