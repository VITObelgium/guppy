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
from dask import array as da
from guppy2.config import config as cfg
from guppy2.db import schemas as s, models as m
from guppy2.raster_calc_utils import create_raster, generate_raster_response, perform_operation, process_raster_list_with_function_in_chunks, rescale_result, insert_into_guppy_db, compare_rasters, \
    convert_raster_to_likeraster


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
            arguments_list.append({'nodata': nodata, 'factor': layer_item.factor, 'operation': layer_item.operation, 'is_rgb': layer_model.is_rgb})
    fixed_path_list = []
    for file in path_list:
        if not compare_rasters(file, path_list[0], check_nodata=False):
            convert_raster_to_likeraster(file, path_list[0], os.path.join(base_path, raster_name))
            fixed_path_list.append(file.replace(".tif", f"_{unique_identifier}.tif"))
        else:
            fixed_path_list.append(file)

    process_raster_list_with_function_in_chunks(fixed_path_list, os.path.join(base_path, raster_name), path_list[0],
                                                function_to_apply=perform_operation, function_arguments={'layer_args': arguments_list, 'output_rgb': body.rgb},
                                                chunks=20, output_bands=4 if body.rgb else 1, dtype=np.uint8 if body.rgb else None, out_nodata=255 if body.rgb else None)
    build_overview_tiles = [2, 4, 8, 16, 32, 64]
    image = gdal.Open(os.path.join(base_path, raster_name), 1)  # 0 = read-only, 1 = read-write.
    gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
    image.BuildOverviews("NEAREST", build_overview_tiles)
    del image
    if body.rescale_result:
        os.rename(src=os.path.join(base_path, raster_name), dst=os.path.join(base_path, raster_name.replace('.tif', 'tmp.tif')))
        with rasterio.open(os.path.join(base_path, raster_name.replace('.tif', 'tmp.tif'))) as ds:
            input_arr = ds.read(out_shape=(int(ds.height / 4), int(ds.width / 4)))
        input_arr = np.where(input_arr == nodata, np.nan, input_arr)
        input_arr = input_arr[~np.isnan(input_arr)]
        if body.rescale_result.rescale_type == s.AllowedRescaleTypes.quantile:
            rescale_result_list = [np.nanquantile(input_arr, b) for b in body.rescale_result.breaks]
            rescale_result_dict = {k: v for k, v in enumerate(rescale_result_list)}
        elif body.rescale_result.rescale_type == s.AllowedRescaleTypes.equal_interval:
            input_arr *= 1.0 / input_arr.max()
            rescale_result_list = body.rescale_result.breaks
            rescale_result_dict = {k: v for k, v in enumerate(rescale_result_list)}
        elif body.rescale_result.rescale_type == s.AllowedRescaleTypes.natural_breaks:
            input_arr *= 1.0 / input_arr.max()
            sample_arr = np.random.choice(input_arr[input_arr != 0], size=10000)  # needs low samples or jenks is too slow
            rescale_result_list = jenkspy.jenks_breaks(sample_arr, n_classes=len(body.rescale_result.breaks))
            rescale_result_dict = {k: v for k, v in enumerate(rescale_result_list)}
        elif body.rescale_result.rescale_type == s.AllowedRescaleTypes.provided:
            rescale_result_dict = body.rescale_result.breaks
        print(rescale_result_dict)
        process_raster_list_with_function_in_chunks([os.path.join(base_path, raster_name.replace('.tif', 'tmp.tif'))], os.path.join(base_path, raster_name),
                                                    os.path.join(base_path, raster_name.replace('.tif', 'tmp.tif')),
                                                    function_to_apply=rescale_result,
                                                    function_arguments={'output_rgb': body.rgb, 'rescale_result_dict': rescale_result_dict, 'nodata': arguments_list[0]['nodata']},
                                                    chunks=20, output_bands=4 if body.rgb else 1, dtype=np.uint8 if body.rgb else None, out_nodata=255 if body.rgb else None)

        build_overview_tiles = [2, 4, 8, 16, 32, 64]
        image = gdal.Open(os.path.join(base_path, raster_name), 1)  # 0 = read-only, 1 = read-write.
        gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
        image.BuildOverviews("NEAREST", build_overview_tiles)
        del image
    print('raster_calculation 200', time.time() - t)
    insert_into_guppy_db(db, raster_name, os.path.join(base_path, raster_name), body.rgb)
    if body.geoserver:
        geoserver_layer = create_raster(raster_name)
        return Response(content=geoserver_layer, status_code=status.HTTP_201_CREATED)
    else:
        return generate_raster_response(os.path.join(base_path, raster_name))


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
