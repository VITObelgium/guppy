import logging
import math
import os

import numpy as np
import rasterio
from fastapi import HTTPException
from rasterio.mask import mask, raster_geometry_mask
from rasterio.windows import from_bounds
from sqlalchemy.orm import Session

from guppy.db import schemas as s
from guppy.db.models import LayerMetadata

logger = logging.getLogger(__name__)
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


def create_quantile_response(rst: np.array, nodata: float, type: str, quantiles: [float], layer_name: str = None) -> s.QuantileResponse:
    """
    Args:
        rst (np.array): The input array containing the raster data.
        mask_array (np.array): The array representing the mask.
        nodata (float): The nodata value in the raster.
        type (str): The type of the response.
        quantiles (List[float]): The quantiles to calculate.
        layer_name (str, optional): The name of the layer.

    Returns:
        s.QuantileResponse: The quantile response object.

    """
    rst = rst.astype(float)
    rst[rst == nodata] = np.nan
    calculated_quantiles = np.nanquantile(rst, quantiles)
    response = s.QuantileResponse(type=type,
                                  quantiles=[s.QuantileList(quantile=quantile, value=no_nan(float(value))) for quantile, value in zip(quantiles, calculated_quantiles)]
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
    Determine the overview level and the corresponding value from the overview levels list based on the resolution and bounds.
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
    Decodes the given data array from rgba to values.

    Args:
        data (ndarray): An array containing the data to decode.

    Returns:
        ndarray: A decoded array.

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
        if file_path and not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        layer_data_chache[layer_name] = file_path
    return layer_data_chache[layer_name]


def sample_coordinates(coords, path, layer_name):
    """
    Args:
        coords (list): The list of coordinates to sample from the raster file.
        path (str): The path to the raster file.
        layer_name (str): The name of the layer.

    Returns:
        s.LineData: An instance of the s.LineData class with the sampled data.

    """
    result = []
    if os.path.exists(path):
        with rasterio.open(path) as src:
            x = src.sample(coords, indexes=1)
            for v in x:
                result.append(v[0])
    else:
        logger.error(f'sample_coordinates: file not found {path}')
    return s.LineData(layer_name=layer_name, data=result)


def sample_coordinates_window(coords_dict, layer_models, bounds, round_val=None):
    """
    Args:
        coords_dict: A dictionary containing the coordinates for each layer. The keys of the dictionary represent the layer names, and the values are lists of coordinate tuples (x, y).
        layer_models: A list of layer models containing information about each layer.
        bounds: A list of four values representing the bounds of the window to sample. The values should be in the order [min x, min y, max x, max y].
        round_val: (optional) The number of decimal places to round the sampled coordinates. If not specified, coordinates will not be rounded.

    Returns:
        result_all: A list of sampled values for each layer.

    """
    result_all = []
    path = layer_models[0].file_path if not layer_models[0].is_mbtile else layer_models[0].data_path
    coords = []
    for k, v in coords_dict.items():
        coords.extend(v)
    with rasterio.open(path) as src:
        geometry_window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], src.transform).round_offsets()
        rows, cols = src.rowcol([p[0] for p in coords], [p[1] for p in coords])
        cols = [c - geometry_window.col_off for c in cols]
        rows = [r - geometry_window.row_off for r in rows]
        in_rows = []
        in_cols = []
        out_idx = []
        in_idx = []
        data = src.read(1, window=geometry_window)
        clipped_data = np.full((max(math.ceil(geometry_window.height), data.shape[0]), max(data.shape[1], math.ceil(geometry_window.width))), fill_value=0, dtype=np.float32)
        if data.shape == clipped_data.shape:
            clipped_data = data
        else:
            # Clip the raster using the geometry
            # Update the portion of clipped_data that overlaps with the geometry
            row_offset = abs(int(geometry_window.row_off)) if geometry_window.row_off < 0 else 0
            col_offset = abs(int(geometry_window.col_off)) if geometry_window.col_off < 0 else 0
            clipped_data[row_offset:row_offset + int(data.shape[0]), col_offset:col_offset + int(data.shape[1])] = data

        # create the metadata for the dataset

        r_max, c_max = clipped_data.shape
        for i, (r, c) in enumerate(zip(rows, cols)):
            if r < 0 or c < 0 or r >= r_max or c >= c_max:
                out_idx.append(i)
            else:
                in_idx.append(i)
                in_rows.append(r)
                in_cols.append(c)

    for layer_model in layer_models:
        result_all.append(sample_layer(in_cols, in_idx, in_rows, layer_model, out_idx, geometry_window, round_val))
    return result_all


def sample_layer(in_cols, in_idx, in_rows, layer_model, out_idx, geometry_window, round_val: int = None):
    """
    Args:
        in_cols: List[int]: List of column indices to extract values from the clipped data.
        in_idx: List[int]: List of indices to assign the extracted values to in the result dictionary.
        in_rows: List[int]: List of row indices to extract values from the clipped data.
        layer_model: LayerModel: The layer model containing the file path and layer name.
        out_idx: List[int]: List of indices to assign the nodata values to in the result dictionary.
        geometry_window: GeometryWindow: The window defining the geometry to clip from the raster.
        round_val: int, optional: The rounding precision for the clipped data. Defaults to None.

    Returns:
        dict: A dictionary containing the layer name and the extracted values based on the given indices.

    """
    path = layer_model.file_path
    with rasterio.open(path) as src:
        data = src.read(1, window=geometry_window)
        nodata = src.nodata
        if nodata is None:
            nodata = -9999

        clipped_data = np.full((max(math.ceil(geometry_window.height), data.shape[0]), max(data.shape[1], math.ceil(geometry_window.width))), fill_value=nodata, dtype=np.float32)
        if data.shape == clipped_data.shape:
            clipped_data = data
        else:
            # Update the portion of clipped_data that overlaps with the geometry
            row_offset = abs(int(geometry_window.row_off)) if geometry_window.row_off < 0 else 0
            col_offset = abs(int(geometry_window.col_off)) if geometry_window.col_off < 0 else 0
            clipped_data[row_offset:row_offset + int(data.shape[0]), col_offset:col_offset + int(data.shape[1])] = data

    result = {}
    if round_val:
        clipped_data = np.round(clipped_data, round_val)
    f = clipped_data[in_rows, in_cols]
    for i, v in zip(in_idx, f):
        result[i] = v
    for i in out_idx:
        result[i] = nodata
    result = [result[key] for key in sorted(result.keys())]
    return {'layerName': layer_model.layer_name, 'data': result}
