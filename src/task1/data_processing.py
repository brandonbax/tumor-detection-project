import geopandas as gpd

import rasterio
from rasterio.features import rasterize
import numpy as np
from PIL import Image
import os
import tifffile
from typing import List, Tuple
import config

# Specs say that all classes (even the background) other than tissue_tumor and tissue_stroma
# must be classified as other in the output.

def get_class_id(row) -> int:
    class_name = row['classification'].get('name')
    # Other classes get the default id of 0
    return config.CLASS_NAME_TO_ID.get(class_name, 0)

def read_tif_image(path: str) -> np.ndarray:
    """
    Read a .tif image and return an RGB uint8 numpy array (H, W, 3).
 
    Handles single-channel, RGBA, channel-first layouts, and
    non-uint8 dtypes transparently.
    """
    try:
        img = tifffile.imread(path)
    except Exception:
        img = np.array(Image.open(path))
 
    # Handle different channel orders / counts
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3:
        if img.shape[0] in (1, 3, 4):          # (C, H, W) → (H, W, C)
            img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 4:                   # RGBA → RGB
            img = img[:, :, :3]
        if img.shape[2] == 1:
            img = np.concatenate([img] * 3, axis=-1)
 
    # Normalise to uint8 if needed
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
 
    return img

def create_mask(label_path: str, image_path: str, output_mask_path: str) -> np.ndarray:
    geo_df = gpd.read_file(label_path)
    # Adds a new column with class id
    geo_df['class_id'] = geo_df.apply(get_class_id, axis=1)

    # Filter out the other classes
    geo_df = geo_df[geo_df['class_id'] != 0]

    with rasterio.open(image_path) as img:
        shapes = ((geometry, int(class_id)) for geometry, class_id in zip(geo_df.geometry, geo_df['class_id']))
        mask = rasterize(shapes=shapes, out_shape=(img.height, img.width),
                         transform=img.transform, fill=0, dtype='uint8')

        # The mask should be saved as a file, so it doesn't have to be recreated and held in memory
        meta = img.meta.copy()
        meta.update({
            'count': 1,
            'dtype': rasterio.uint8,
            'nodata': 0
        })

        with rasterio.open(output_mask_path, 'w', **meta) as dst:
            dst.write(mask, 1)

    return mask

def match_image_label_pairs(label_path: str, image_path: str) -> List[Tuple[str, str]]:
    image_ext = ".tif"
    label_ext = ".geojson"
 
    images = {}
    for f in os.listdir(image_path):
        if f.endswith(image_ext):
            stem = os.path.splitext(f)[0]
            images[stem] = os.path.join(image_path, f)
 
    labels = {}
    for f in os.listdir(label_path):
        if f.endswith(label_ext):
            stem = os.path.splitext(f)[0]
            labels[stem] = os.path.join(label_path, f)
 
    pairs = []
    for stem in sorted(images.keys()):
        if stem in labels:
            pairs.append((images[stem], labels[stem]))
        else:
            print(f"No label found for image: {images[stem]}")

    return pairs

def compute_dataset_stats(image_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    means, stds = [], []
    for f in os.listdir(image_dir):
        if f.endswith('.tif'):
            path = os.path.join(image_dir, f)
            img = read_tif_image(path).astype(np.float32) / 255.0
            means.append(img.mean(axis=(0, 1)))
            stds.append(img.std(axis=(0, 1)))
    return np.mean(means, axis=0), np.mean(stds, axis=0)

if __name__ == "__main__":
    print("Dataset stats:")
    means, stds = compute_dataset_stats(config.TRAIN_IMAGE_DIR)
    print("Means:", means)
    print("Stds:", stds)
