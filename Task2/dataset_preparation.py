"""
Task 2: Dataset Preparation

Approach:
1. Goes through image and geojson pairs in train and val splits.
2. For each nucleus, computes the centroid using GeoJSON coordinates and crops a 100x100px patch.
3. Randomly samples exactly N_TRAIN_PER_CLASS / N_VAL_PER_CLASS patches per class.
4. Builds a contrastive set from the remaining train patches (no overlap with train set).
5. Saves patches and labels as .npy files and class mapping as JSON.

- Why 100x100px patch?
  The Task 2 test set files are already 100×100 uint8 patches (verified during data exploration).
"""

import json
import random
from pathlib import Path
import numpy as np
from PIL import Image

DATASET_PATH = Path(__file__).parent.parent / "Dataset_Splits"
OUTPUT_DIR = Path(__file__).parent / "task2_dataset"

PATCH_SIZE = 100
HALF = PATCH_SIZE // 2

# As per CW spec: only these 3 classes are used for Task 2
TARGET_CLASSES = [
    "nuclei_histiocyte",
    "nuclei_lymphocyte",
    "nuclei_tumor",
]

# As per CW spec: 2500 train / 700 val patches per class
N_TRAIN_PER_CLASS = 2500
N_VAL_PER_CLASS   = 700

def load_image_as_rgb(image_path):
    img = Image.open(image_path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def polygon_centroid(coordinates):
    xs = [pt[0] for pt in coordinates]
    ys = [pt[1] for pt in coordinates]
    # Centroid by arithmetic mean of polygon vertices:
    # cx = round((1/N) * sum_i x_i), cy = round((1/N) * sum_i y_i)
    cx = int(round(sum(xs) / len(xs)))
    cy = int(round(sum(ys) / len(ys)))
    return cx, cy

def extract_patch(image, cx, cy, patch_size=PATCH_SIZE):
    # Image size: H = number of rows (y), W = number of columns (x)
    H, W = image.shape[:2]
    # half = floor(patch_size / 2)
    half = patch_size // 2

    # Source crop window centered at (cx, cy):
    # x0 = cx - half, y0 = cy - half, x1 = x0 + patch_size, y1 = y0 + patch_size
    x0_src = cx - half
    y0_src = cy - half
    x1_src = x0_src + patch_size
    y1_src = y0_src + patch_size

    # Clip source window to valid image bounds [0, W] x [0, H]
    x0_img = max(x0_src, 0)
    y0_img = max(y0_src, 0)
    x1_img = min(x1_src, W)
    y1_img = min(y1_src, H)

    # Map clipped image window into destination patch coordinates
    x0_dst = x0_img - x0_src
    y0_dst = y0_img - y0_src
    x1_dst = x0_dst + (x1_img - x0_img)
    y1_dst = y0_dst + (y1_img - y0_img)

    # Zero-padded patch; copy overlap region from image into patch
    patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    patch[y0_dst:y1_dst, x0_dst:x1_dst] = image[y0_img:y1_img, x0_img:x1_img]
    return patch

def build_class_index(class_names):
    return {name: i for i, name in enumerate(class_names)}


def extract_all_patches(split_dir, class_to_idx):
    """Extract all patches for target classes; returns dict {class_name: [patches]}."""
    image_dir  = split_dir / "image"
    nuclei_dir = split_dir / "nuclei"

    per_class = {name: [] for name in class_to_idx}

    for geojson_path in sorted(nuclei_dir.glob("*.geojson")):
        stem       = geojson_path.stem
        image_path = image_dir / (stem.replace("_nuclei", "") + ".tif")

        if not image_path.exists():
            print(f"  [WARNING] Image not found, skipping: {image_path.name}")
            continue

        image = load_image_as_rgb(image_path)

        with open(geojson_path) as f:
            features = json.load(f).get("features", [])

        for feature in features:
            class_name = (feature.get("properties", {})
                                 .get("classification", {})
                                 .get("name", None))

            if class_name not in class_to_idx:
                continue

            geom_type = feature["geometry"]["type"]
            if geom_type == "Polygon":
                ring_coords = feature["geometry"]["coordinates"][0]
            elif geom_type == "MultiPolygon":
                ring_coords = feature["geometry"]["coordinates"][0][0]
            else:
                continue

            cx, cy = polygon_centroid(ring_coords)
            patch  = extract_patch(image, cx, cy, patch_size=PATCH_SIZE)
            per_class[class_name].append(patch)

    return per_class


def sample_patches(per_class, n_per_class, seed=42):
    """Randomly sample n_per_class patches from each class. Returns (patches, int_labels, str_labels, remainder)."""
    class_to_idx  = build_class_index(list(per_class.keys()))
    patches, int_labels, str_labels = [], [], []
    remainder = {}  # patches not used in this sample (for contrastive set)

    rng = random.Random(seed)
    for class_name, all_patches in per_class.items():
        shuffled = all_patches[:]
        rng.shuffle(shuffled)
        chosen   = shuffled[:n_per_class]
        leftover = shuffled[n_per_class:]

        if len(chosen) < n_per_class:
            print(f"  [WARNING] {class_name}: only {len(chosen)} patches "
                  f"available, requested {n_per_class}")

        for p in chosen:
            patches.append(p)
            int_labels.append(class_to_idx[class_name])
            str_labels.append(class_name)

        remainder[class_name] = leftover

    return patches, int_labels, str_labels, remainder


def save_split(patches, int_labels, string_labels, split_name, output_dir):
    # Save the arrays for one split to disk.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = np.stack(patches, axis=0).astype(np.uint8) 
    y = np.array(int_labels, dtype=np.int32)         
    y_names = np.array(string_labels, dtype=object)

    np.save(output_dir / f"X_{split_name}.npy",       X)
    np.save(output_dir / f"y_{split_name}.npy",       y)
    np.save(output_dir / f"y_names_{split_name}.npy", y_names)

    print(f"  Saved: X_{split_name}.npy {X.shape}, "
          f"y_{split_name}.npy {y.shape}")

    # Print per class counts for a quick sanity check
    unique, counts = np.unique(y_names, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"    {cls}: {cnt}")


def print_class_distribution(split_name, y_names):
    """Print a simple class count table for one split."""
    unique, counts = np.unique(y_names, return_counts=True)
    total = len(y_names)
    print(f"\n  Class distribution — {split_name} ({total} total):")
    for cls, cnt in zip(unique, counts):
        print(f"    {cls:<30s} {cnt:>6d}  ({100*cnt/total:5.1f}%)")


def main():
    print("=" * 60)
    print("Nuclei Patch Dataset Preparation")
    print("=" * 60)

    class_to_idx = build_class_index(TARGET_CLASSES)
    print(f"\nClass mapping:")
    for name, idx in class_to_idx.items():
        print(f"  {idx}: {name}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "class_names.json", "w") as f:
        json.dump({str(v): k for k, v in class_to_idx.items()}, f, indent=2)
    print(f"Class names saved to: {OUTPUT_DIR / 'class_names.json'}")

    # --- Train split ---
    print(f"\nProcessing split: train")
    train_per_class = extract_all_patches(DATASET_PATH / "train", class_to_idx)
    for cls, ps in train_per_class.items():
        print(f"  {cls}: {len(ps)} available")

    train_patches, train_int, train_str, contrastive_pool = sample_patches(
        train_per_class, N_TRAIN_PER_CLASS
    )
    print_class_distribution("train", train_str)
    save_split(train_patches, train_int, train_str, "train", OUTPUT_DIR)

    # --- Contrastive set (remaining patches after train sample, no leakage) ---
    contrast_patches, contrast_int, contrast_str = [], [], []
    for class_name, leftover in contrastive_pool.items():
        idx = class_to_idx[class_name]
        for p in leftover:
            contrast_patches.append(p)
            contrast_int.append(idx)
            contrast_str.append(class_name)

    print_class_distribution("contrastive", contrast_str)
    save_split(contrast_patches, contrast_int, contrast_str, "contrastive", OUTPUT_DIR)

    # --- Validation split ---
    print(f"\nProcessing split: validation")
    val_per_class = extract_all_patches(DATASET_PATH / "validation", class_to_idx)
    for cls, ps in val_per_class.items():
        print(f"  {cls}: {len(ps)} available")

    val_patches, val_int, val_str, _ = sample_patches(
        val_per_class, N_VAL_PER_CLASS
    )
    print_class_distribution("validation", val_str)
    save_split(val_patches, val_int, val_str, "validation", OUTPUT_DIR)

    print("\nDataset preparation complete.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
