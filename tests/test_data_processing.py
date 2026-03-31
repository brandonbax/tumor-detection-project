import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine
from shapely.geometry import Polygon

import task1.data_processing as dp


@pytest.fixture
def temp_workspace():
    """Creates a temporary directory that automatically deletes itself after tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def dummy_tif(temp_workspace):
    """Creates a tiny 10x10 pixel TIF file."""
    tif_path = temp_workspace / "dummy_image.tif"

    # We use an Identity transform (1 pixel = 1 coordinate unit)
    # This makes testing math incredibly easy. Top-left is (0,0).
    transform = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    meta = {
        'driver': 'GTiff', 'height': 10, 'width': 10, 'count': 3,  # 3 channels for RGB
        'dtype': rasterio.uint8, 'transform': transform
    }

    with rasterio.open(tif_path, 'w', **meta) as dst:
        # Write blank data
        dst.write(np.zeros((3, 10, 10), dtype=rasterio.uint8))

    return tif_path


@pytest.fixture
def dummy_geojson(temp_workspace):
    """Creates a GeoJSON with a known tissue_tumor square and an Other square."""
    geojson_path = temp_workspace / "dummy_labels.geojson"

    # Square 1: A 2x2 box from coordinates (2,2) to (4,4)
    tumor_poly = Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])

    # Square 2: A 2x2 box from coordinates (7,7) to (9,9)
    # We will label this tissue_white_background so it should be classed as 'Other'
    glass_poly = Polygon([(7, 7), (9, 7), (9, 9), (7, 9)])

    gdf = gpd.GeoDataFrame({
        'classification': [{'name': 'tissue_tumor'}, {'name': 'tissue_white_background'}],
        'geometry': [tumor_poly, glass_poly]
    })

    gdf.to_file(geojson_path, driver='GeoJSON')
    return geojson_path


def test_mask_creation_outputs_correct_file(dummy_geojson, dummy_tif, temp_workspace):
    """Tests if the file is physically created and has the correct metadata."""
    out_path = temp_workspace / "output_mask.tif"

    dp.create_mask(dummy_geojson, dummy_tif, out_path)

    # Assert the file was created
    assert out_path.exists()

    # Assert the file metadata is correct (1 channel, 10x10)
    with rasterio.open(out_path) as src:
        assert src.count == 1
        assert src.width == 10
        assert src.height == 10
        assert src.dtypes[0] == 'uint8'


def test_mask_creation_rasterizes_correctly(dummy_geojson, dummy_tif, temp_workspace):
    """Tests if the vector polygons were burned into the correct pixels."""
    out_path = temp_workspace / "output_mask.tif"

    mask_array = dp.create_mask(dummy_geojson, dummy_tif, out_path)

    # Assert the background is 0 (e.g. pixel at 0,0)
    assert mask_array[0, 0] == 0

    # Assert the tissue_tumor was rasterized to '1'
    # Our polygon was from x:2-4, y:2-4. Let's check the middle of it.
    assert mask_array[3, 3] == 1

    # Assert the 'tissue_white_background' was correctly classed as 'Other' (remains 0)
    # Our polygon was from x:7-9, y:7-9.
    assert mask_array[8, 8] == 0

    # Count the total tissue_tumor pixels. A 2x2 coordinate square covers exactly 4 pixels.
    assert np.sum(mask_array == 1) == 4
