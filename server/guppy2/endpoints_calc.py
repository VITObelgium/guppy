import io
import os
import time

import rasterio
import numpy as np
from pyproj import Transformer
from shapely import wkt
from shapely.ops import transform
from sqlalchemy.orm import Session
from starlette import status
from starlette.responses import Response
import requests
from guppy2.error import create_error
from fastapi.responses import StreamingResponse
from rasterio.enums import Resampling
from guppy2.config import config as cfg
from guppy2.db import schemas as s, models as m
from guppy2.endpoint_utils import _extract_area_from_dataset, _decode


def create_raster(input_arr, crs, r_transform, dtype=None, nodata=-9999, geoserver=False):
    t = time.time()
    mem = io.BytesIO()
    geoserver_layer = None
    if np.all(np.mod(input_arr, 1) == 0):
        input_arr = np.array(input_arr, dtype=np.int32)
    with rasterio.open(mem, 'w', driver='GTiff',
                       height=input_arr.shape[0], width=input_arr.shape[1],
                       count=1, dtype=str(input_arr.dtype) if dtype is None else dtype,
                       crs=crs,
                       nodata=nodata,
                       transform=r_transform,
                       tiled=True, compress='deflate') as dst:
        dst.write(input_arr, 1)
        dst.build_overviews([2, 4, 8, 16, 32, 64, 128], Resampling.nearest)
        dst.update_tags(ns='rio_overview', resampling='nearest')
        nan_arr = input_arr[input_arr != nodata]
        dst.update_tags(minimum=np.nanmin(nan_arr), maximum=np.nanmax(nan_arr), source='guppy.calculate')
    mem.seek(0)
    print("done geotiff generation", time.time() - t)
    if geoserver:
        t = time.time()
        base_path = 'content/tifs/generated'
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        with open(os.path.join(base_path, 'test.tif'), "wb") as file:
            file.write(mem.getvalue())
        geoserver_layer = create_geoserver_layer('test.tif', 'generated_tif')
        print("done geoserver", time.time() - t)
    mem.seek(0)
    return mem, geoserver_layer


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
    if generated_file is None:
        return create_error('no result generated', 204)
    return StreamingResponse(generated_file, media_type="image/tiff")


def perform_operation(base_arr, input_arr, nodata, operation: s.AllowedOperations, factor):
    if base_arr is None:
        base_arr = input_arr.copy() * factor
    else:
        if operation == s.AllowedOperations.multiply:
            base_arr = np.where(base_arr == nodata, base_arr, base_arr * np.where(input_arr == nodata, input_arr, input_arr * factor))
        elif operation == s.AllowedOperations.add:
            base_arr = np.where(base_arr == nodata, base_arr, base_arr + np.where(input_arr == nodata, 0, input_arr * factor))
        elif operation == s.AllowedOperations.subtract:
            base_arr = np.where(base_arr == nodata, base_arr, base_arr - np.where(input_arr == nodata, 0, input_arr * factor))
    return base_arr


def raster_calculation(db: Session, body: s.RasterCalculationBody):
    print('raster_calculation')
    t = time.time()
    output_arr = None
    crs = None
    r_transform = None
    nodata = None
    for layer_item in body.layer_list:
        layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_item.layer_name).first()
        if layer_model:
            path = layer_model.file_path
            if os.path.exists(path) and body:
                with rasterio.open(path) as src:
                    rst_arr = src.read(1)
                    if crs is None:
                        crs = src.crs
                    if nodata is None:
                        nodata = src.nodata
                    if r_transform is None:
                        r_transform = src.transform
                    if layer_model.is_rgb:
                        rst_arr = _decode(rst_arr)
                output_arr = perform_operation(output_arr, rst_arr, nodata, layer_item.operation, layer_item.factor)
            else:
                print('WARNING: file does not exists', path)
    if output_arr is None:
        return Response(content='layer not found', status_code=status.HTTP_404_NOT_FOUND)
    generated_file, geoserver_layer = create_raster(output_arr, crs, r_transform, dtype=None, nodata=nodata, geoserver=body.geoserver)
    print('raster_calculation 200', time.time() - t)
    if body.geoserver:
        return Response(content=geoserver_layer, status_code=status.HTTP_201_CREATED)
    else:
        return generate_raster_response(generated_file)
