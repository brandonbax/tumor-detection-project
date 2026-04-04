import geopandas as gpd

import rasterio
from rasterio.features import rasterize

# Specs say that all classes (even the background) other than tissue_tumor and tissue_stroma
# must be classified as other in the output.

class_ids = {
    'tissue_tumor': 1,
    'tissue_stroma': 2,
}

def get_class_id(row) -> int:
    class_name = row['classification'].get('name')
    # Other classes get the default id of 0
    return class_ids.get(class_name, 0)

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
