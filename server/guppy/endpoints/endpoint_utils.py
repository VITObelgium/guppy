import logging
import math
import os

import numpy as np
import rasterio
from fastapi import HTTPException
from rasterio.mask import mask, raster_geometry_mask
from rasterio.windows import from_bounds
from rasterio.transform import rowcol
from sqlalchemy.orm import Session
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
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


def create_stats_response_polygon(path, geom, layer_model, overview_factor:int,layer_name:str=None):
    """
    Create a statistics response based on the polygon method for small raster datasets.

    This function calculates various statistical metrics (e.g., min, max, mean, quantiles)
    based on the intersection of a polygon with a raster dataset. It processes raster data
    within the specified region, determines weighted statistics, and packages the results
    in a response object.

    Args:
        path (str): The file path to the raster dataset.
        geom (dict): The geometry defining the area of interest, as a GeoJSON-like dict.
        layer_model (LayerModel): A model representing the layer's metadata, including
            color information and data properties.
        overview_factor (int): The level of the overview to use for reading raster data.
        layer_name (str, optional): The name of the layer associated with the statistics.

    Returns:
        StatsResponse: A response object containing calculated statistics for the raster
            data in the provided polygon area.

    Raises:
        ValueError: If required input parameters are not provided or invalid.
        RasterioIOError: If the raster file cannot be opened or processed.
        GeoPandasError: If the intersection geometries cannot be computed.
    """
    logger.info("fallback to polygon method for small raster in stats")
    with rasterio.open(path, overview_level=overview_factor) as src:
        rst, crop_transform = _extract_area_from_dataset(src, [geom], crop=True, all_touched=True, is_rgb=layer_model.is_rgb)
        if layer_model.is_rgb:
            rst = _decode(rst)
        shape_mask = _extract_shape_mask_from_dataset(src, [geom], all_touched=True, crop=True)
        pixel_res = abs(src.res[0] * src.res[1])
        nodata = src.nodata if src.nodata is not None else -9999
        crs = src.crs.to_epsg()
    transform_to_use = crop_transform if crop_transform is not None else src.transform
    mask = np.where(shape_mask == 0, rst, nodata)

    polygon_shapes = []
    for geom_shape, value in shapes(mask.astype(np.float32), transform=transform_to_use):
        if value != nodata:
            poly = shape(geom_shape)
            polygon_shapes.append({'geometry': poly, 'value': value})

    if polygon_shapes:
        raster_gdf = gpd.GeoDataFrame(polygon_shapes, crs=f"EPSG:{crs}")
        input_gdf = gpd.GeoDataFrame([{'geometry': geom}], crs=f"EPSG:{crs}")
        intersections = gpd.overlay(raster_gdf, input_gdf, how='intersection', keep_geom_type=False)
        if not intersections.empty:
            intersections['area'] = intersections.geometry.area

            values = intersections['value'].values
            areas = intersections['area'].values

            valid_mask = values != nodata
            values = values[valid_mask]
            areas = areas[valid_mask]

            if len(values) > 0:
                weighted_mean = np.sum(values * areas) / np.sum(areas)
                min_val = float(np.min(values))
                max_val = float(np.max(values))
                sum_val = float(np.sum(values * areas / pixel_res))

                pixel_counts = np.maximum(np.round(areas).astype(int), 1)
                weighted_samples = []
                for val, count in zip(values, pixel_counts):
                    weighted_samples.extend([val] * count)
                weighted_samples = np.array(weighted_samples, dtype=float)

                q2, q5, q95, q98 = np.quantile(weighted_samples, [0.02, 0.05, 0.95, 0.98])

                count_data = np.sum(np.isfinite(rst)&shape_mask == 0)
                count_total = np.sum(shape_mask == 0)  # Including nodata polygons
                count_no_data = count_total - count_data

                response = s.StatsResponse(type="polygon stats",
                                           min=no_nan(min_val),
                                           max=no_nan(max_val),
                                           sum=no_nan(sum_val),
                                           mean=no_nan(weighted_mean),
                                           count_no_data=count_no_data,
                                           count_total=count_total,
                                           count_data=count_data,
                                           q02=no_nan(float(q2)),
                                           q05=no_nan(float(q5)),
                                           q95=no_nan(float(q95)),
                                           q98=no_nan(float(q98)),
                                           )
                if layer_name:
                    response.layer_name = layer_name
                return response

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


def _calculate_classification_polygon_method(rst, shape_mask, input_geom, src, crop_transform=None):
    """
    Calculates classification results based on polygon method for small raster data.

    The function processes raster data and extracts classification polygons based on
    the provided shape mask and input geometry. It computes the intersection
    between the resulting polygon shapes and the input geometry to determine the
    corresponding classification statistics such as value, count, and percentage
    distribution.

    Args:
        rst: np.ndarray. The raster data array for processing.
        shape_mask: np.ndarray. Binary mask defining the region of interest in the raster.
        input_geom: shapely.geometry.base.BaseGeometry. Input geometry against which
            classification polygons are intersected.
        src: rasterio.io.DatasetReader. Source raster dataset providing
            metadata (e.g., nodata value, CRS).
        crop_transform: affine.Affine, optional. Transformation matrix to apply to
            the cropped raster; defaults to the transformation of the source raster if not provided.

    Returns:
        s.ClassificationResult: An instance encapsulating the classification results.
        It contains the list of classified entries, including the value, count, and
        calculated percentage.
    """
    logger.info("fallback to polygon method for small raster")
    transform = crop_transform if crop_transform is not None else src.transform

    mask = np.where(shape_mask == 0, rst, src.nodata if src.nodata is not None else -9999)

    polygon_shapes = []
    for geom_shape, value in shapes(mask.astype(np.int32), transform=transform):
        if value != (src.nodata if src.nodata is not None else -9999):
            poly = shape(geom_shape)
            polygon_shapes.append({'geometry': poly, 'value': value})

    if not polygon_shapes:
        return s.ClassificationResult(type='classification', data=[])

    raster_gdf = gpd.GeoDataFrame(polygon_shapes, crs=f"EPSG:{src.crs.to_epsg()}")
    input_gdf = gpd.GeoDataFrame([{'geometry': input_geom}], crs=f"EPSG:{src.crs.to_epsg()}")

    intersections = gpd.overlay(raster_gdf, input_gdf, how='intersection', keep_geom_type=False)

    if intersections.empty:
        return s.ClassificationResult(type='classification', data=[])

    intersections['area'] = intersections.geometry.area
    total_area = input_gdf.area.values[0]

    value_stats = intersections.groupby('value').agg({        'area': 'sum'    }).reset_index()

    result_classes = []
    for _, row in value_stats.iterrows():
        value = row['value']
        area = row['area']

        percentage = (area / total_area) * 100 if total_area > 0 else 0

        result_classes.append({
            'value': int(value),
            'count': -1,
            'percentage': percentage
        })

    final_classes = []
    for class_data in result_classes:
        final_classes.append(s.ClassificationEntry(
            value=class_data['value'],
            count=class_data['count'],
            percentage=class_data['percentage']
        ))

    return s.ClassificationResult(type='classification_polygon', data=final_classes)

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


def validate_layer_and_get_file_path(db: Session, layer_name: str, file_type = None) -> str:
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
        if file_type and not file_path.endswith(file_type):
            file_path = layer.data_path
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
    if os.path.exists(path[1:]):
        with rasterio.open(path[1:]) as src:
            x = src.sample(coords, indexes=1)
            for v in x:
                result.append(v[0])
    else:
        logger.error(f'sample_coordinates: file not found {path}')
    return {'layerName': layer_name, 'data': result}


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
    with rasterio.open(path[1:]) as src:
        geometry_window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], src.transform).round_offsets()
        rows, cols = rowcol(src.transform, [p[0] for p in coords], [p[1] for p in coords])
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
    path = layer_model.file_path[1:]
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
