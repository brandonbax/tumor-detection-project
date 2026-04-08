import os
import warnings
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.features import rasterize
from PIL import Image
import tifffile

import config

# Could probably use an alternative to rasterio, since its not particularly
# made for medical imaging, but it works if the file format warnings are suppressed.
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

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
    """
    Create a mask from a label file and an image file.
    
    Args:
        label_path: Path to the label file.
        image_path: Path to the image file.
        output_mask_path: Path to save the mask.
    
    Returns:
        np.ndarray: The mask as a numpy array.
    """
    # Return cached mask if it already exists on disk
    if os.path.exists(output_mask_path):
        with rasterio.open(output_mask_path) as src:
            return src.read(1)

    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)

    geo_df = gpd.read_file(label_path)
    # Adds a new column with class id
    geo_df['class_id'] = geo_df.apply(get_class_id, axis=1)

    # Filter out the other classes
    geo_df = geo_df[geo_df['class_id'] != 0]

    with rasterio.open(image_path) as img:
        shapes = ((geometry, int(class_id)) for geometry, class_id in zip(geo_df.geometry, geo_df['class_id']))
        mask = rasterize(shapes=shapes, out_shape=(img.height, img.width),
                         transform=img.transform, fill=0, dtype='uint8')

        # Save the mask so it doesn't have to be recreated on subsequent runs
        meta = img.meta.copy()
        meta.update({
            'count': 1,
            'dtype': rasterio.uint8,
            'nodata': 0
        })

        with rasterio.open(output_mask_path, 'w', **meta) as dst:
            dst.write(mask, 1)

    return mask

def match_image_label_pairs(image_path: str, label_path: str) -> List[Tuple[str, str]]:
    """
    Match image and label pairs.
    
    Args:
        image_path: Path to the image directory.
        label_path: Path to the label directory.
    
    Returns:
        List[Tuple[str, str]]: List of (image_path, label_path) pairs.
    """
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
            # Strip the _tissue suffix so stems match image filenames
            if stem.endswith("_tissue"):
                stem = stem[:-len("_tissue")]
            labels[stem] = os.path.join(label_path, f)
 
    pairs = []
    for stem in sorted(images.keys()):
        if stem in labels:
            pairs.append((images[stem], labels[stem]))
        else:
            print(f"No label found for image: {images[stem]}")

    return pairs

# The dataset stats are used in the config file for normalisation.
# If the dataset is changed, these stats should be updated.
def compute_dataset_stats(image_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean and standard deviation of the dataset.
    
    Args:
        image_dir: Path to the image directory.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of (means, stds).
    """
    means, stds = [], []
    for f in os.listdir(image_dir):
        if f.endswith('.tif'):
            path = os.path.join(image_dir, f)
            img = read_tif_image(path).astype(np.float32) / 255.0
            means.append(img.mean(axis=(0, 1)))
            stds.append(img.std(axis=(0, 1)))
    return np.mean(means, axis=0), np.mean(stds, axis=0)

if __name__ == "__main__":
    """
    Compute the mean and standard deviation of the dataset.
    """
    print("Dataset stats:")
    means, stds = compute_dataset_stats(config.TRAIN_IMAGE_DIR)
    print("Means:", means)
    print("Stds:", stds)
