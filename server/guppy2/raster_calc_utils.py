import os
import time
from typing import Callable
from sqlalchemy.orm import Session
import numpy as np
import rasterio
import requests
import rioxarray as xr
from dask import array as da
from osgeo import osr, gdal
from starlette.responses import StreamingResponse

from guppy2.config import config as cfg
from guppy2.db import schemas as s
from guppy2.db import models as m
from guppy2.endpoint_utils import _decode
from guppy2.error import create_error
from guppy2.rasterio_file_streamer import RIOFile


def create_raster(raster_name):
    t = time.time()
    try:
        geoserver_layer = create_geoserver_layer(raster_name, raster_name.split('.')[0])
        print("done geoserver", time.time() - t)
        return geoserver_layer
    except requests.exceptions.ConnectionError as e:
        create_error(message='geoserver caused error')


def insert_into_guppy_db(db: Session, filename, file_path, is_rgb):
    layer_name = filename.split('.')[-1]
    new_layer = m.LayerMetadata(layer_name=f"{layer_name}_store:{layer_name}", file_path=file_path, is_rgb=is_rgb)
    db.add(new_layer)
    db.commit()


def create_geoserver_layer(data_source, layer_name):
    # Set up authentication (if required)
    username = cfg.geoserver.username
    password = cfg.geoserver.password
    auth = (username, password)

    workspace = "generated"
    coverage_store = f"{layer_name}_store"

    json_data = {
        "coverageStore": {
            "name": coverage_store,
            "workspace": {
                "name": workspace
            },
            "type": "GeoTIFF",
            "enabled": True,
            "url": r'file:///content/tifs/generated/' + data_source  # Provide the path to the raster data source here
        },
    }

    base_url = "http://geoserver:8080/geoserver/rest/"
    # base_url = "https://guppy2.marvintest.vito.be/geoserver/rest/"
    headers = {"Content-Type": "application/json"}
    url = f"{base_url}workspaces/{workspace}/coveragestores"

    response = requests.post(url, json=json_data, auth=auth, headers=headers)

    if response.status_code == 201:
        print("Raster store created successfully.")
    else:
        print(f"Failed to create raster store. Status code: {response.status_code}")
        print(response.text)
    json_data = {
        "coverage": {
            "description": "Generated tif",
            "enabled": True,
            "name": layer_name,
            "nativeFormat": "GeoTIFF",
            "title": "generated tif"
        }
    }

    url = f"{base_url}workspaces/{workspace}/coveragestores/{coverage_store}/coverages"
    response = requests.post(url, json=json_data, auth=auth, headers=headers)

    if response.status_code == 201:
        print("layer created successfully.")
    else:
        print(f"Failed to create layer. Status code: {response.status_code}")
        print(response.text)
    return f"{coverage_store}:{layer_name}"


def generate_raster_response(generated_file):
    def generate_geotiff():
        with open(generated_file, "rb") as geotiff_file:
            while chunk := geotiff_file.read(8192):
                yield chunk

    return StreamingResponse(generate_geotiff(), media_type="image/tiff")


def perform_operation(*input_arrs, layer_args, output_rgb):
    output_arr = None
    first = True
    out_nodata = -9999
    for input_arr, args_dict in zip(input_arrs, layer_args):
        nodata = args_dict['nodata']
        factor = args_dict['factor']
        operation = args_dict['operation']
        is_rgb = args_dict['is_rgb']
        if is_rgb:
            input_arr = _decode(input_arr)
        if first:
            output_arr = np.where(input_arr == nodata, input_arr, input_arr * factor)
            out_nodata = nodata
            first = False
        else:
            if operation == s.AllowedOperations.multiply:
                output_arr = np.where(output_arr == nodata, output_arr, output_arr * np.where(input_arr == nodata, 1, input_arr * factor))
            elif operation == s.AllowedOperations.add:
                output_arr = np.where(output_arr == nodata, output_arr, output_arr + np.where(input_arr == nodata, 0, input_arr * factor))
            elif operation == s.AllowedOperations.subtract:
                output_arr = np.where(output_arr == nodata, output_arr, output_arr - np.where(input_arr == nodata, 0, input_arr * factor))
            elif operation == s.AllowedOperations.boolean_mask:
                output_arr = np.where(output_arr == nodata, output_arr, output_arr * np.where(input_arr == nodata, 1, input_arr))
            elif operation == s.AllowedOperations.invert_boolean_mask:
                output_arr = np.where(output_arr == nodata, output_arr, output_arr * np.where(input_arr == nodata, 1, 1 - input_arr))
    if output_rgb:
        output_arr = data_to_rgba(output_arr, out_nodata)
    return output_arr


def rescale_result(*input_arrs, output_rgb, rescale_result_list=None, nodata=None):
    rescaled_output_arr = None
    if nodata is None:
        nodata = -9999
    for input_arr in input_arrs:
        if output_rgb:
            input_arr = _decode(input_arr)
        output_arr = np.where(input_arr == nodata, np.nan, input_arr)
        min_val = 0
        for index, value in enumerate(rescale_result_list):
            max_val = value
            rescaled_output_arr = np.where((min_val <= output_arr) & (output_arr < max_val), index * 1 / len(rescale_result_list), output_arr)
    if output_rgb:
        rescaled_output_arr = data_to_rgba(rescaled_output_arr, nodata)
    return rescaled_output_arr


def data_to_rgba(data, nodata):
    data = np.squeeze(data)
    data = data.astype(np.float32)
    if np.isnan(nodata):
        data = np.where(np.isnan(data), -9999, data)
        nodata = -9999
    rows, cols = data.shape
    rgb = np.frombuffer(data.astype('<f4').tobytes(), dtype=np.uint8).reshape(-1, 4).transpose().reshape(4, rows, cols).copy()

    rgb[3] = np.where(data == nodata, 255, rgb[3])
    rgb[2] = np.where(data == nodata, 255, rgb[2])
    rgb[1] = np.where(data == nodata, 255, rgb[1])
    rgb[0] = np.where(data == nodata, 255, rgb[0])

    return rgb


def _get_raster_epsg(input_raster_ds):
    proj = osr.SpatialReference(wkt=input_raster_ds.GetProjection())
    proj.AutoIdentifyEPSG()
    return int(proj.GetAttrValue('AUTHORITY', 1))


def _get_raster_bounds(input_raster_ds):
    """ returns the bounds and resolution of a given raster dataset

    Args:
        input_raster_ds: gdal dataset of a raster

    Returns:
        x_min, y_min, x_max, y_max, resolution
    """
    gt = input_raster_ds.GetGeoTransform()
    x_min = gt[0]
    y_max = gt[3]
    resolution_x = gt[1]
    resolution_y = -gt[5]
    y_min = y_max - (resolution_y * input_raster_ds.RasterYSize)
    x_max = x_min + (resolution_x * input_raster_ds.RasterXSize)
    return x_min, y_min, x_max, y_max, resolution_x, resolution_y


def compare_rasters(raster_path_a: str, raster_path_b: str, check_nodata: bool = True):
    """ compares raster a with raster b. retruns True if equal.
        checks on geotransform, size, nodata value, epsg code and bounds.

    Args:
        raster_path_a: path to file
        raster_path_b: path to file
        check_nodata: boolean to skip nodata check

    Returns:
        Boolean: True or False depending on equality.
    """
    a_ds = gdal.Open(raster_path_a)
    b_ds = gdal.Open(raster_path_b)
    band_a = a_ds.GetRasterBand(1)
    band_b = b_ds.GetRasterBand(1)

    if a_ds.GetGeoTransform() != b_ds.GetGeoTransform():
        print(f'geotransform not equal: {a_ds.GetGeoTransform()} != {b_ds.GetGeoTransform()}')
        return False
    if a_ds.RasterXSize != b_ds.RasterXSize or a_ds.RasterYSize != b_ds.RasterYSize:
        print(f'size not equal: {a_ds.RasterXSize},{a_ds.RasterYSize} != {b_ds.RasterXSize},{b_ds.RasterYSize} ')
        return False
    if check_nodata and band_a.GetNoDataValue() != band_b.GetNoDataValue():
        print(f'nodata not equal: {band_a.GetNoDataValue()} != {band_b.GetNoDataValue()}')
        return False
    if _get_raster_epsg(a_ds) != _get_raster_epsg(b_ds):
        print(f'epsg not equal: {_get_raster_epsg(a_ds)} != {_get_raster_epsg(b_ds)}')
        return False
    if _get_raster_bounds(a_ds) != _get_raster_bounds(b_ds):
        print(f'bounds not equal: {_get_raster_bounds(a_ds)} != {_get_raster_bounds(b_ds)}')
        return False
    return True


def process_raster_list_with_function_in_chunks(input_file_list: [str], output_file: str, like_file: str, function_to_apply: Callable[..., np.ndarray], function_arguments: {} = None,
                                                chunks: int = 10,
                                                overlap_cells: tuple = 0,
                                                dtype=None, output_bands=1, out_nodata=None):
    """ Function to process large rasters in smaller parts to reduce memory footprint and enable parallelization.

     Args:
         input_file_list: list of paths to input raster files
         output_file: path to output raster file
         like_file: raster file to use as output format
         function_to_apply: function to apply on the raster. Has to have two ndarray as the first two parameter, and must return a ndarray.
         function_arguments: arguments to be given to the function_to_apply
         chunks: number of horizontal and vertical tiles to make
         overlap_cells: number of cells that the individual tiles share with their neighbors. needs to be (1,10,10) if you want 10 cell overlap for single band rasters.
         dtype: output array dtype.If None, dask tries to guess it from a limited set.
     Throws:
        AssertionError: if input_file_1 and input_file_2 are not equal (size, resolution, nodata value, epsg) this exception is raised.
     """
    t = time.time()
    if function_arguments is None:
        function_arguments = {}
    if not os.path.exists(os.path.dirname(output_file)) and os.path.dirname(output_file):
        os.mkdir(os.path.dirname(output_file))
    if len(input_file_list) > 1:
        for input_file in input_file_list[1:]:
            if not compare_rasters(input_file_list[0], input_file, check_nodata=False):
                raise AssertionError(f"Input raster {input_file} is not comparable to the first one. Please align them first with convert_raster_to_likeraster()")
    input_da_arrays = []
    open_da_arrays = []
    ds = rasterio.open(like_file)
    chunk_size = (1, int(ds.shape[0] / chunks), int(ds.shape[1] / chunks))
    for input_file in input_file_list:
        values_da_arr = xr.open_rasterio(input_file, chunks=chunk_size)
        input_da_arrays.append(values_da_arr.data)
        open_da_arrays.append(values_da_arr)  # keep reference of all open dask arrays to close them at the end to free file handles
    r = da.map_overlap(function_to_apply, *input_da_arrays, depth=overlap_cells, boundary='reflect', trim=True,
                       align_arrays=True, dtype=dtype, **function_arguments)
    profile = ds.profile
    profile.update(
        driver='GTiff',
        count=output_bands,
        tiled=True,
        compress='deflate',
        num_threads='ALL_CPUS',
        width=ds.shape[1],
        height=ds.shape[0],
        blockxsize=256,
        blockysize=256,
        bigtiff='YES')
    if dtype == np.float32:
        profile.update(dtype=rasterio.float32)
    elif dtype is not None:
        profile.update(dtype=dtype)
    if out_nodata is not None:
        profile.update(nodata=out_nodata)
    with RIOFile(output_file, 'w', **profile) as r_file:
        da.store(r, r_file, lock=True)
    for open_da in open_da_arrays:
        open_da.close()
    print(f"combine_rasters_with_function_in_chunks done, {time.time() - t}")
