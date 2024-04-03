import math
import os

import numpy as np
import rasterio
from fastapi import HTTPException
from rasterio.mask import mask, raster_geometry_mask
from sqlalchemy.orm import Session

from guppy2.db import schemas as s
from guppy2.db.models import LayerMetadata

layer_data_chache = {}


def get_overview_factor(bounds, native, path):
    """
    Args:
        bounds: A tuple or list containing the bounding box coordinates of the area of interest.
        native: A boolean flag indicating whether to use the native resolution of the raster file or not. If set to False, the overview resolution will be used instead.
        path: A string representing the file path of the raster file.

    Returns:
        overview_factor: An integer representing the overview factor to be used for the raster file. This value determines the level of downsampling or simplification of the data.
        overview_bin: An integer representing the bin size for the overview factor. This value determines the size of each cell or bin in the overview.
    """
    if not native:
        with rasterio.open(path) as src:
            res_x, res_y = src.res
            overview_factor, overview_bin = get_overview(res_x, res_y, src.overviews(1), bounds)
    else:
        overview_factor, overview_bin = None, None
    return overview_factor, overview_bin


def no_nan(input):
    """
    Checks if the input is NaN (Not a Number) and returns None if it is.

    Args:
        input: The value to be checked.

    Returns:
        The input value if it is not NaN, None otherwise.
    """
    if math.isnan(input):
        return None
    return input


def create_stats_response(rst: np.array, mask_array: np.array, nodata: float, type: str, layer_name: str = None):
    """
    Args:
        rst (np.array): The input array containing the raster data.
        mask_array (np.array): The array representing the mask.
        nodata (float): The nodata value in the raster.
        type (str): The type of the response.
        layer_name (str, optional): The name of the layer.

    Returns:
        s.StatsResponse: The stats response object.

    """
    rst = rst.astype(float)
    rst[rst == nodata] = np.nan
    q2, q5, q95, q98 = np.nanquantile(rst, [0.02, 0.05, 0.95, 0.98])
    response = s.StatsResponse(type=type,
                               min=no_nan(float(np.nanmin(rst))),
                               max=no_nan(float(np.nanmax(rst))),
                               sum=no_nan(float(np.nansum(rst))),
                               mean=no_nan(float(np.nanmean(rst))),
                               count_no_data=int(np.sum((~np.isfinite(rst)) & (~mask_array))),
                               count_total=int(np.sum(~mask_array)),
                               count_data=int(np.sum(np.isfinite(rst))),
                               q02=no_nan(float(q2)),
                               q05=no_nan(float(q5)),
                               q95=no_nan(float(q95)),
                               q98=no_nan(float(q98)),
                               )
    if layer_name:
        response.layer_name = layer_name
    return response


def _extract_area_from_dataset(raster_ds, geom, crop=True, all_touched=False, is_rgb=False):
    """
    Args:
        raster_ds: A raster dataset from which to extract the area.
        geom: The geometry representing the area to be extracted.
        crop: A boolean indicating whether to crop the extracted area to the exact boundaries of the geometry. Default is True.
        all_touched: A boolean indicating whether to include pixels that are partially covered by the geometry. Default is False.
        is_rgb: A boolean indicating whether the raster dataset contains RGB values. Default is False.

    Returns:
        If is_rgb is True, returns a tuple containing the cropped array and the transform. If is_rgb is False, returns a tuple containing the cropped array (with shape (1, height, width
    *)) and the transform.
    """
    crop_arr, crop_transform = mask(raster_ds, geom, crop=crop, all_touched=all_touched)
    if is_rgb:
        return crop_arr, crop_transform
    crop_arr = crop_arr[0]
    return crop_arr, crop_transform


def _extract_shape_mask_from_dataset(raster_ds, geom, crop=True, all_touched=False):
    """
    Args:
        raster_ds: The raster dataset from which to extract the shape mask.
        geom: The geometry representing the shape to extract.
        crop: A boolean indicating whether or not to crop the shape mask to the extent of the input geometry. Default is True.
        all_touched: A boolean indicating whether to include all pixels touching the shape, or only those that are fully covered by the shape. Default is False.

    Returns:
        shape_mask: A binary mask where pixels within the shape are True and pixels outside the shape are False.
    """
    shape_mask, transform, window = raster_geometry_mask(raster_ds, geom, all_touched=all_touched, crop=crop)
    return shape_mask


def get_overview(res_x: float, res_y: float, overviews: [int], bounds: (float,)):
    """
    Args:
        res_x (float): The resolution of the x-axis.
        res_y (float): The resolution of the y-axis.
        overviews (List[int]): The available overview levels.
        bounds (Tuple[float]): The bounds of the area of interest.

    Returns:
        Tuple[int, int]: The overview level and the corresponding value from the overview levels list.

    """
    bbox_bottom, bbox_left, bbox_top, bbox_right = bounds
    pixels = (bbox_right - bbox_left) / res_x * (bbox_top - bbox_bottom) / res_y
    factor = int(pixels / 4000000)
    overview_level = 0
    for i, value in enumerate(overviews):
        if value > factor:
            overview_level = i - 1
            break
    if overview_level < 0:
        return None, None
    return overview_level, overviews[overview_level]


def _decode(data):
    """
    Utility to decode RGB encoded data
    """
    return np.frombuffer(data.reshape(4, -1).transpose().tobytes(), dtype='<f4').reshape((data[0].shape))


def validate_layer_and_get_file_path(db: Session, layer_name: str) -> str:
    """
    Args:
        db: The database session to use for querying the LayerMetadata table.
        layer_name: The name of the layer to validate and get the file path for.

    Returns:
        The file path of the specified layer.

    Raises:
        HTTPException: If the layer cannot be found in the database or if the file path does not exist.
    """
    if layer_name not in layer_data_chache:
        layer = db.query(LayerMetadata).filter_by(layer_name=layer_name).first()
        if not layer:
            raise HTTPException(status_code=404, detail=f"Layer not found: {layer_name}")
        file_path = layer.file_path
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        layer_data_chache[layer_name] = file_path
    return layer_data_chache[layer_name]
