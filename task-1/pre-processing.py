import geopandas as gpd
from collections import defaultdict

class_map = defaultdict({
    'tissue_tumor': 1,
    'tissue_stroma': 2
})

def process_file(file_dir):
    geo_df = gpd.read_file(file_dir)
    # for i, row in enumerate(geo_df['classification']):
    #     geo_df['classification'][i] = class_map(row['name'])
    print(geo_df['classification'])
    print(geo_df['classification'][0]['name'])

process_file("Dataset_Splits/train/tissue/training_set_primary_roi_089_tissue.geojson")