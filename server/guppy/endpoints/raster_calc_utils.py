import itertools
import logging
import os
import time
from typing import Callable

import numpy as np
import rasterio
import requests
import rioxarray as xr
from dask import array as da
from osgeo import osr, gdal
from sqlalchemy.orm import Session
from starlette import status
from starlette.responses import StreamingResponse

from guppy.config import config as cfg
from guppy.db import models as m
from guppy.db import schemas as s
from guppy.endpoints.endpoint_utils import _decode
from guppy.endpoints.rasterio_file_streamer import RIOFile
from guppy.error import create_error

logger = logging.getLogger(__name__)


def create_raster(raster_name, style=None):
    t = time.time()
    try:
        geoserver_layer = create_geoserver_layer(raster_name, raster_name.split('.')[0], style)
        logger.info(f"done geoserver {time.time() - t}")
        return geoserver_layer
    except requests.exceptions.ConnectionError as e:
        create_error(message='geoserver caused error')


def insert_into_guppy_db(db: Session, filename, label, file_path, is_rgb):
    layer_name = filename.split('.')[0]
    new_layer = m.LayerMetadata(layer_name=f"generated:{layer_name}", label=label, file_path=file_path, is_rgb=is_rgb)
    db.add(new_layer)
    db.commit()


def create_geoserver_layer(data_source, layer_name, sld_name=None):
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
            "url": rf'file://{cfg.deploy.content}/tifs/generated/' + data_source  # Provide the path to the raster data source here
        },
    }

    base_url = "http://geoserver:8080/geoserver/rest/"
    headers = {"Content-Type": "application/json"}
    url = f"{base_url}workspaces/{workspace}/coveragestores"

    response = requests.post(url, json=json_data, auth=auth, headers=headers)

    if response.status_code == 201:
        logger.info("Raster store created successfully.")
    else:
        logger.warning(f"Failed to create raster store. Status code: {response.status_code}")
        logger.info(response.text)
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
        logger.info("layer created successfully.")
    else:
        logger.warning(f"Failed to create layer. Status code: {response.status_code}")
        logger.info(response.text)
    if sld_name and len(str(sld_name).strip()) > 0 and str(sld_name).strip() != 'nan':
        url = base_url + f'workspaces/{workspace}/layers/{layer_name}.xml'
        headers = {'content-type': 'text/xml', 'Accept-Charset': 'UTF-8'}
        data = f'<layer><defaultStyle><name>{sld_name}</name></defaultStyle></layer>'
        r = requests.put(url, data=data, headers=headers, auth=auth)
        if not r.ok:
            logger.warning(f"Failed to set style. {layer_name}")
    return f"{workspace}:{layer_name}"


def generate_raster_response(generated_file):
    def generate_geotiff():
        with open(generated_file, "rb") as geotiff_file:
            while chunk := geotiff_file.read(8192):
                yield chunk

    return StreamingResponse(generate_geotiff(), media_type="image/tiff")


def perform_operation(*input_arrs, layer_args, output_rgb, unique_values=None):
    out_nodata = -9999
    output_arr = None
    unique_values = unique_values or [None] * len(input_arrs)  # Ensure unique_values has the same length as input_arrs

    for idx, (input_arr, args_dict, unique_vals) in enumerate(zip(input_arrs, layer_args, unique_values)):
        factor = args_dict['factor']
        operation = args_dict['operation']
        is_rgb = args_dict['is_rgb']
        if is_rgb:
            input_arr = _decode(input_arr)
        if input_arr.dtype == np.byte or input_arr.dtype == np.uint8 or input_arr.dtype == np.uint16 or input_arr.dtype == np.uint32 or input_arr.dtype == np.int8 or input_arr.dtype == np.int16:
            input_arr = input_arr.astype(np.int32)
        input_arr[input_arr == args_dict['nodata']] = out_nodata
        input_arr[np.isnan(input_arr)] = out_nodata
        if idx == 0:
            output_arr = np.where(input_arr != out_nodata, input_arr * factor, input_arr)
            out_unique = unique_vals
            output_arr_nodata = output_arr != out_nodata
        else:
            if (operation == s.AllowedOperations.multiply
                    or operation == s.AllowedOperations.add
                    or operation == s.AllowedOperations.subtract):
                mask_nodata = input_arr != out_nodata  # Compute mask once to reuse it
                input_arr_masked = np.full_like(input_arr, 0)
                np.multiply(input_arr, factor, out=input_arr_masked, where=mask_nodata)  # Factor multiplication again, but now with the mask
            if operation == s.AllowedOperations.multiply:
                np.multiply(output_arr, np.where(mask_nodata, input_arr_masked, 1), out=output_arr, where=(output_arr_nodata) & (input_arr != out_nodata))
            elif operation == s.AllowedOperations.add:
                np.add(output_arr, input_arr_masked, out=output_arr, where=(output_arr_nodata) & (input_arr != out_nodata))
            elif operation == s.AllowedOperations.subtract:
                np.subtract(output_arr, input_arr_masked, out=output_arr, where=output_arr_nodata)
            elif operation == s.AllowedOperations.boolean_mask:
                np.multiply(output_arr, input_arr, out=output_arr, where=(output_arr_nodata) & (input_arr != out_nodata))
            elif operation == s.AllowedOperations.clip:
                output_arr[input_arr != 1] = out_nodata
            elif operation == s.AllowedOperations.invert_boolean_mask:
                np.multiply(output_arr, 1 - input_arr, out=output_arr, where=(output_arr_nodata) & (input_arr != out_nodata))
            elif operation == s.AllowedOperations.unique_product:
                combo_arr = output_arr.copy()
                for idx, (u1, u2) in enumerate(itertools.product(out_unique, unique_vals)):
                    mask = (output_arr == u1) & (input_arr == u2)
                    combo_arr[mask] = idx
                output_arr[:] = combo_arr  # Update output_arr in-place
            elif operation == s.AllowedOperations.normalize:
                valid_mask = output_arr != out_nodata
                valid_min = np.nanmin(output_arr[valid_mask])
                valid_max = np.nanmax(output_arr[valid_mask])
                output_arr = np.where(valid_mask, (output_arr - valid_min) / (valid_max - valid_min), out_nodata)
    if output_rgb:
        output_arr = data_to_rgba(output_arr, out_nodata)
    return output_arr


def apply_rescale_result(*input_arrs, output_rgb, rescale_result_dict=None, nodata=None, bins=False, normalize=None):
    if nodata is None:
        nodata = -9999
    rescaled_output_arr = np.full_like(input_arrs[0], nodata)
    for input_arr in input_arrs:
        if output_rgb:
            input_arr = _decode(input_arr)
        if normalize is not None:
            input_arr_norm = input_arr * (1.0 / normalize)
        else:
            input_arr_norm = input_arr
        if bins:
            bin_borders = [value for key, value in rescale_result_dict.items()]
            rescaled_output_arr = np.where((np.isnan(input_arr)) | (input_arr == nodata), nodata, np.digitize(input_arr_norm, bins=bin_borders, right=True))
            for i, key in enumerate([key for key, value in rescale_result_dict.items()]):
                rescaled_output_arr = np.where(rescaled_output_arr == i, key, rescaled_output_arr)
        else:
            for key, value in rescale_result_dict.items():
                if '-' in str(value) and not str(value).startswith('-'):
                    min_val = float(value.split('-')[0])
                    max_val = float(value.split('-')[1])
                    rescaled_output_arr[(not np.isnan(input_arr)) & (input_arr != nodata) & (min_val <= input_arr_norm) & (input_arr_norm < max_val)] = int(key)
                else:
                    if isinstance(value, list):
                        rescaled_output_arr[np.isin(input_arr_norm, value)] = int(key)
                    else:
                        rescaled_output_arr[input_arr_norm == value] = int(key)
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
        logger.warning(f'geotransform not equal: {a_ds.GetGeoTransform()} != {b_ds.GetGeoTransform()}')
        return False
    if a_ds.RasterXSize != b_ds.RasterXSize or a_ds.RasterYSize != b_ds.RasterYSize:
        logger.warning(f'size not equal: {a_ds.RasterXSize},{a_ds.RasterYSize} != {b_ds.RasterXSize},{b_ds.RasterYSize} ')
        return False
    if check_nodata and band_a.GetNoDataValue() != band_b.GetNoDataValue():
        logger.warning(f'nodata not equal: {band_a.GetNoDataValue()} != {band_b.GetNoDataValue()}')
        return False
    if _get_raster_epsg(a_ds) != _get_raster_epsg(b_ds):
        logger.warning(f'epsg not equal: {_get_raster_epsg(a_ds)} != {_get_raster_epsg(b_ds)}')
        return False
    if _get_raster_bounds(a_ds) != _get_raster_bounds(b_ds):
        logger.warning(f'bounds not equal: {_get_raster_bounds(a_ds)} != {_get_raster_bounds(b_ds)}')
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
                raise AssertionError(f"Input raster {input_file} is not comparable to {input_file_list[0]}. Please align them first with convert_raster_to_likeraster()")
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
    input_da_arrays = None
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
    input_da_arrays = None
    logger.info(f"combine_rasters_with_function_in_chunks done, {time.time() - t}")


def warp_raster(input_raster_file: str, output_raster_file: str, resolution: float, output_bounds: [], target_epsg: int, nodata: float, resampling=gdal.GRA_Average,
                as_array: bool = False, error_threshold: float = 0.125, options: [] = None,
                build_overviews: bool = False, resolution_y: float = None, source_epsg: int = None):
    """
    Args:
        input_raster_file: path to input file
        output_raster_file: path to output file
        resolution: the target resolution (float)
        output_bounds: [x_min, y_min, x_max, y_max]
        target_epsg: output epsg code (int)
        nodata: This value will be used for cells withoyt data.
        resampling: resampling algorithm (default gdal.GRA_Average)
        as_array: return numpy array (default False)
        error_threshold: Error threshold for transformation approximation (in pixel units - defaults to 0.125)
        options: These options will be passed without modification to the 'options' argument of the gdal.Warp() function. (dflt:  ['COMPRESS=LZW', "TILED=YES"])
        build_overviews: If not None and not empty, the overview-tiles of the given resample values will be created.
        resolution_y: in case x and y resolution are not the same.
        source_epsg: input epsg code (int, default None)
    Returns:
        numpy array (ONLY if parameter as_array == True)
    """
    if options is None:
        options = ['COMPRESS=DEFLATE', "TILED=YES", 'BIGTIFF=YES', 'NUM_THREADS=ALL_CPUS']
    output_raster_file = str(output_raster_file)
    kwargs = {}
    input_raster = gdal.Open(str(input_raster_file))
    raster_srs = osr.SpatialReference()
    if target_epsg is None:
        raster_srs.ImportFromWkt(input_raster.GetProjectionRef())
    else:
        raster_srs.ImportFromEPSG(int(target_epsg))
    kwargs['dstSRS'] = raster_srs
    source_srs = osr.SpatialReference()
    if source_epsg is None:
        source_srs.ImportFromWkt(input_raster.GetProjectionRef())
    else:
        source_srs.ImportFromEPSG(int(source_epsg))
    kwargs['srcSRS'] = source_srs

    if output_bounds is not None:
        kwargs['outputBoundsSRS'] = raster_srs
        kwargs['outputBounds'] = output_bounds
    if resolution is not None:
        kwargs['xRes'] = resolution
        if resolution_y is not None:
            kwargs['yRes'] = resolution_y
        else:
            kwargs['yRes'] = resolution
    if error_threshold is not None:
        kwargs['errorThreshold'] = error_threshold
    if resampling is not None:
        kwargs['resampleAlg'] = resampling
    if nodata is not None:
        kwargs['dstNodata'] = nodata
    if options is not None:
        kwargs['creationOptions'] = options

    kwargs['format'] = 'GTiff'
    kwargs['multithread'] = True
    kwargs['warpOptions'] = ['NUM_THREADS=ALL_CPUS']

    output_raster = gdal.Warp(output_raster_file, input_raster, **kwargs)
    logger.info("warp done")

    return_array = output_raster.ReadAsArray() if as_array else []
    output_raster = None
    if build_overviews:
        build_overview_tiles = [2, 4, 8, 16, 32, 64]
        image = gdal.Open(output_raster_file, 1)  # 0 = read-only, 1 = read-write.
        gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
        image.BuildOverviews("AVERAGE", build_overview_tiles)
        del image

    return return_array


def convert_raster_to_likeraster(input_raster_file: str, like_raster_file: str, output_file: str, error_threshold: float = 0.125, resampling=gdal.GRA_Average):
    """convert raster to match the likeraster given

    Args:
        input_raster_file: path to file
        like_raster_file: path to file
        output_file: path to output file
        error_threshold: error_threshold used for the resampling. default:  0.125
        resampling: type of resampling to use. default: gdal.GRA_Average
    """
    input_ds = gdal.Open(input_raster_file)
    like_ds = gdal.Open(like_raster_file)
    band = like_ds.GetRasterBand(1)
    x_min, y_min, x_max, y_max, resolution_x, resolution_y = _get_raster_bounds(like_ds)
    nodata = band.GetNoDataValue()
    output_bounds = [x_min, y_min, x_max, y_max]
    target_epsg = _get_raster_epsg(like_ds)
    warp_raster(input_raster_file, output_file, resolution_x, output_bounds, target_epsg, nodata, resampling=resampling, error_threshold=error_threshold, resolution_y=resolution_y)
    logger.info("done conversion")


def cleanup_files(path_list, unique_identifier):
    for path in path_list:
        if path.endswith(f'{unique_identifier}_fixed.tif') and os.path.exists(path):
            os.remove(path)


def get_unique_values(arguments_list, fixed_path_list):
    unique_values = []
    if s.AllowedOperations.unique_product in [arg['operation'] for arg in arguments_list]:
        for path, arg in zip(fixed_path_list, arguments_list):
            if arg['operation_data']:
                unique_values.append(arg['operation_data'])
            else:
                unique_values_set = set()
                with rasterio.open(path) as src:
                    for ji, window in src.block_windows():
                        arr = src.read(window=window)
                        unique_values_set.update(np.unique(arr))
                unique_values.append(list(unique_values_set))
    return unique_values


def align_files(base_path, path_list, unique_identifier):
    fixed_path_list = []
    for file in path_list:
        if not compare_rasters(file, path_list[0], check_nodata=False):
            if not os.path.exists(os.path.join(base_path, file.replace(".tif", f"_{unique_identifier}_fixed.tif"))):
                convert_raster_to_likeraster(file, path_list[0], file.replace(".tif", f"_{unique_identifier}_fixed.tif"), resampling=gdal.GRA_NearestNeighbour)
            fixed_path_list.append(file.replace(".tif", f"_{unique_identifier}_fixed.tif"))
        else:
            fixed_path_list.append(file)
    return fixed_path_list


def fill_path_and_argument_lists(arguments_list, layer_list, db, nodata, path_list):
    for layer_item in layer_list:
        layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_item.layer_name).first()
        if layer_model:
            path = layer_model.file_path if not layer_model.is_mbtile else layer_model.data_path
            path_list.append(path)
            if os.path.exists(path):
                with rasterio.open(path) as src:
                    if nodata is None:
                        nodata = src.nodata
            else:
                logger.info(f'WARNING: file does not exists {path}')
                create_error(message='layer not found', code=status.HTTP_404_NOT_FOUND)
            arguments_list.append({'nodata': nodata, 'factor': layer_item.factor, 'operation': layer_item.operation, 'operation_data': layer_item.operation_data, 'is_rgb': layer_model.is_rgb})
        else:
            logger.info(f'WARNING: layer_model does not exists {layer_item.layer_name}')


def read_raster_without_nodata_as_array(path: str) -> np.ndarray:
    output = []
    with rasterio.open(path) as ds:
        for ji, window in ds.block_windows(1):
            data = ds.read(1, window=window)
            data = data[data != ds.nodata]
            output.append(data)

    return np.concatenate(output)

