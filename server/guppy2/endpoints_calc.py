import datetime
import os
import random
import time

import jenkspy
import numpy as np
import rasterio
import requests
from osgeo import gdal
from sqlalchemy.orm import Session
from starlette import status
from starlette.responses import Response

from guppy2.config import config as cfg
from guppy2.db import schemas as s, models as m
from guppy2.raster_calc_utils import create_raster, generate_raster_response, perform_operation, process_raster_list_with_function_in_chunks, apply_rescale_result, insert_into_guppy_db, cleanup_files, \
    get_unique_values, align_files


def raster_calculation(db: Session, body: s.RasterCalculationBody):
    print('raster_calculation')
    t = time.time()
    base_path = '/content/tifs/generated'
    unique_identifier = f'{datetime.datetime.now().strftime("%Y-%m-%d")}_{str(random.randint(0, 10000000))}'
    raster_name = f'generated_{unique_identifier}.tif'
    nodata = None
    first = True
    path_list = []
    arguments_list = []
    for layer_item in body.layer_list:
        layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_item.layer_name).first()
        if layer_model:
            path = layer_model.file_path
            path_list.append(path)
            if os.path.exists(path) and body:
                with rasterio.open(path) as src:
                    if nodata is None:
                        nodata = src.nodata
            else:
                print('WARNING: file does not exists', path)
                return Response(content='layer not found', status_code=status.HTTP_404_NOT_FOUND)
            arguments_list.append({'nodata': nodata, 'factor': layer_item.factor, 'operation': layer_item.operation, 'operation_data': layer_item.operation_data, 'is_rgb': layer_model.is_rgb})
        else:
            print('WARNING: layer_model does not exists', layer_item.layer_name)
    fixed_path_list = align_files(base_path, path_list, unique_identifier)
    unique_values = get_unique_values(arguments_list, fixed_path_list)
    print('perform_operation', time.time() - t, unique_values)
    process_raster_list_with_function_in_chunks(fixed_path_list, os.path.join(base_path, raster_name), fixed_path_list[0],
                                                function_to_apply=perform_operation, function_arguments={'layer_args': arguments_list, 'output_rgb': body.rgb, 'unique_values': unique_values},
                                                chunks=10, output_bands=4 if body.rgb else 1, dtype=np.uint8 if body.rgb else None, out_nodata=255 if body.rgb else -9999)
    if body.rescale_result:
        process_rescaling(arguments_list, base_path, body, nodata, raster_name, t)
    build_overview_tiles = [2, 4, 8, 16, 32, 64]
    image = gdal.Open(os.path.join(base_path, raster_name), 1)  # 0 = read-only, 1 = read-write.
    gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
    image.BuildOverviews("NEAREST", build_overview_tiles)
    del image
    cleanup_files(fixed_path_list, unique_identifier)
    print('raster_calculation 200', time.time() - t)
    insert_into_guppy_db(db, raster_name, os.path.join(base_path, raster_name), body.rgb)
    if body.geoserver:
        geoserver_layer = create_raster(raster_name, body.result_style)
        return Response(content=geoserver_layer, status_code=status.HTTP_201_CREATED)
    else:
        return generate_raster_response(os.path.join(base_path, raster_name))


def process_rescaling(arguments_list, base_path, body, nodata, raster_name, t):
    bins = False
    normalize = None
    os.rename(src=os.path.join(base_path, raster_name), dst=os.path.join(base_path, raster_name.replace('.tif', 'tmp.tif')))
    if body.rescale_result.rescale_type != s.AllowedRescaleTypes.provided:
        with rasterio.open(os.path.join(base_path, raster_name.replace('.tif', 'tmp.tif'))) as ds:
            input_arr = ds.read(1)
        input_arr = np.where(input_arr == nodata, np.nan, input_arr)
        input_arr = input_arr[~np.isnan(input_arr)]
        if body.rescale_result.rescale_type == s.AllowedRescaleTypes.quantile:
            rescale_result_list = [np.nanquantile(input_arr, b) for b in body.rescale_result.breaks]
            rescale_result_dict = {k: v for k, v in enumerate(rescale_result_list)}
            bins = True
        elif body.rescale_result.rescale_type == s.AllowedRescaleTypes.equal_interval:
            normalize = input_arr.max()
            input_arr *= 1.0 / normalize
            rescale_result_list = body.rescale_result.breaks
            rescale_result_dict = {k: v for k, v in enumerate(rescale_result_list)}
            bins = True
        elif body.rescale_result.rescale_type == s.AllowedRescaleTypes.natural_breaks:
            normalize = input_arr.max()
            input_arr *= 1.0 / normalize
            sample_arr = np.random.choice(input_arr[input_arr != 0], size=10000)  # needs low samples or jenks is too slow
            rescale_result_list = jenkspy.jenks_breaks(sample_arr, n_classes=len(body.rescale_result.breaks) - 1)
            rescale_result_dict = {k: v for k, v in enumerate(rescale_result_list)}
            sample_arr = None
            bins = True
        input_arr = None
    else:
        rescale_result_dict = body.rescale_result.breaks
    print(rescale_result_dict, bins)
    print('rescale_result', time.time() - t)
    if nodata.is_integer():
        nodata = int(nodata)
    process_raster_list_with_function_in_chunks([os.path.join(base_path, raster_name.replace('.tif', 'tmp.tif'))], os.path.join(base_path, raster_name),
                                                os.path.join(base_path, raster_name.replace('.tif', 'tmp.tif')),
                                                function_to_apply=apply_rescale_result,
                                                function_arguments={'output_rgb': body.rgb, 'rescale_result_dict': rescale_result_dict, 'nodata': nodata, 'bins': bins, 'normalize': normalize},
                                                chunks=10, output_bands=4 if body.rgb else 1, dtype=np.uint8 if body.rgb else rasterio.int32 if bins else None, out_nodata=255 if body.rgb else nodata)


def delete_generated_store(layer_name):
    print('delete_generated_store')
    username = cfg.geoserver.username
    password = cfg.geoserver.password
    auth = (username, password)

    workspace = "generated"
    coverage_store = layer_name.split(":")[0]
    base_path = '/content/tifs/generated'

    base_url = "http://geoserver:8080/geoserver/rest/"
    # base_url = "https://guppy2.marvintest.vito.be/geoserver/rest/"
    headers = {"Content-Type": "application/json"}
    url = f"{base_url}workspaces/{workspace}/coveragestores/{coverage_store}?purge=all&recurse=true"

    response = requests.delete(url, auth=auth, headers=headers)

    if response.status_code == 200:
        print(f"Raster store {coverage_store} deleted successfully.")
        os.remove(os.path.join(base_path, f'{layer_name.split(":")[1]}.tif'))
    else:
        print(f"Failed to delete raster store {coverage_store}. Status code: {response.status_code}")
        print(response.text)
    return response
