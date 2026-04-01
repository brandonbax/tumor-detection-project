"""
Task 2: Dataset Preparation

Approach:
1. Goes through image and geojson pairs in train, val and test splits.
2. For each nucleus, it computer the centroid using GeoJSON coordinates and crops it into a 100x100px patch. Determine class using properties["classifciation]["name"]
3. Saves the patches and labels as .npy files and classes in a json. (index: class_name)

- Why choosing a 100x100px patch?

The Task 2 test set files are already in 100×100 uint8 patches (verified during data exploration)
So we deicded to use the same size across train, validation and test splits.
"""

# Imports
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Paths
DATASET_PATH = Path(__file__).parent.parent / "Dataset_Splits"
OUTPUT_DIR = Path(__file__).parent / "task2_dataset"

# Patch Size to extract nucleus, chose 100 as it matches the test set 
PATCH_SIZE = 100

# pixels on each side of the centroid
HALF = PATCH_SIZE // 2

# Discovered by scanning all GeoJSON files across every split during data exploration.
# Tissue classes are excluded for task 2
KNOWN_CLASSES = [
    "nuclei_apoptosis",
    "nuclei_endothelium",
    "nuclei_epithelium",
    "nuclei_histiocyte",
    "nuclei_lymphocyte",
    "nuclei_melanophage",
    "nuclei_neutrophil",
    "nuclei_plasma_cell",
    "nuclei_stroma",
    "nuclei_tumor",
]

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

def process_split(split_dir, class_to_idx, split_name):
    image_dir  = split_dir / "image"
    nuclei_dir = split_dir / "nuclei"

    # Collect all nuclei GeoJSON files in this split
    geojson_files = sorted(nuclei_dir.glob("*.geojson"))

    patches       = []
    int_labels    = []
    string_labels = []

    for geojson_path in geojson_files:
        # Derive the matching image filename.
        # GeoJSON name: training_set_metastatic_roi_001_nuclei.geojson
        # Image name  : training_set_metastatic_roi_001.tif
        stem = geojson_path.stem                    # "...001_nuclei"
        image_stem = stem.replace("_nuclei", "")    # "...001"
        image_path = image_dir / (image_stem + ".tif")

        if not image_path.exists():
            print(f"  [WARNING] Image not found, skipping: {image_path.name}")
            continue

        # Load the full tissue image
        image = load_image_as_rgb(image_path)

        # Load nuclei annotations
        with open(geojson_path) as f:
            geojson_data = json.load(f)

        features = geojson_data.get("features", [])
        for feature in features:
            # Extract class label from GeoJSON properties
            props = feature.get("properties", {})
            classification = props.get("classification", {})
            class_name = classification.get("name", None)

            if class_name is None:
                # Skip unannotated nuclei
                continue

            # Only keep classes we know about; warn about unexpected ones
            if class_name not in class_to_idx:
                print(f"  [WARNING] Unknown class '{class_name}', skipping.")
                continue

            # Compute centroid of the polygon ring.
            # Handle Polygon and MultiPolygon geometry types.
            geom_type = feature["geometry"]["type"]
            if geom_type == "Polygon":
                # coordinates[0] is the exterior ring; [1:] are holes.
                ring_coords = feature["geometry"]["coordinates"][0]
            elif geom_type == "MultiPolygon":
                # Use the first polygon's exterior ring.
                ring_coords = feature["geometry"]["coordinates"][0][0]
            else:
                # Skip unexpected geometry types (e.g. Point, LineString).
                print(f"  [WARNING] Unsupported geometry type '{geom_type}', "
                      f"skipping.")
                continue

            cx, cy = polygon_centroid(ring_coords)

            # Extract the 100×100 patch centred at (cx, cy)
            patch = extract_patch(image, cx, cy, patch_size=PATCH_SIZE)

            patches.append(patch)
            int_labels.append(class_to_idx[class_name])
            string_labels.append(class_name)

    print(f"  {split_name}: {len(patches)} patches extracted from "
          f"{len(geojson_files)} GeoJSON files.")
    return patches, int_labels, string_labels


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

    # Build a fixed class→integer mapping
    class_to_idx = build_class_index(KNOWN_CLASSES)
    print(f"\nClass mapping:")
    for name, idx in class_to_idx.items():
        print(f"  {idx}: {name}")

    # Save the class mapping to JSON so other scripts can load it
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "class_names.json", "w") as f:
        # Store as {index_str: class_name} for easy JSON loading
        json.dump({str(v): k for k, v in class_to_idx.items()}, f, indent=2)
    print(f"Class names saved to: {OUTPUT_DIR / 'class_names.json'}")

    # Process each split
    splits = ["train", "validation", "test"]
    for split in splits:
        split_dir = DATASET_PATH / split
        if not split_dir.exists():
            print(f"\n[SKIP] Split directory not found: {split_dir}")
            continue

        print(f"Processing split: {split}")
        patches, int_labels, string_labels = process_split(
            split_dir, class_to_idx, split
        )

        if len(patches) == 0:
            print(f"  [WARNING] No patches extracted for split '{split}'.")
            continue

        print_class_distribution(split, string_labels)
        save_split(patches, int_labels, string_labels, split, OUTPUT_DIR)

    print("Dataset preparation complete.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
