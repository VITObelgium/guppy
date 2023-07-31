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

from error import create_error
from fastapi.responses import StreamingResponse

from guppy2.config import config as cfg
from guppy2.db import schemas as s, models as m
from guppy2.endpoint_utils import _extract_area_from_dataset, _decode


def create_raster(input_arr_arr, crs, r_transform, dtype=None, nodata=-9999):
    mem = io.BytesIO()
    with rasterio.open(mem, 'w', driver='GTiff',
                       height=input_arr_arr.shape[0], width=input_arr_arr.shape[1],
                       count=1, dtype=str(input_arr_arr.dtype) if dtype is None else dtype,
                       crs=crs,
                       nodata=nodata,
                       transform=r_transform,
                       tiled=True, compress='deflate') as dst:
        dst.write(input_arr_arr, 1)
        nan_arr = input_arr_arr[input_arr_arr != nodata]
        dst.update_tags(minimum=np.nanmin(nan_arr), maximum=np.nanmax(nan_arr), source='guppy.calculate')
    mem.seek(0)
    return mem


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
    t = time.time()
    output_arr = None
    crs = None
    r_transform = None
    nodata = None
    for layer_item in body.layer_list:
        layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_item.layer_name).first()
        if layer_model:
            path = layer_model.file_path[1:]
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
    generated_file = create_raster(output_arr, crs, r_transform, dtype=None, nodata=nodata)
    print('raster_calculation 200', time.time() - t)
    return generate_raster_response(generated_file)
