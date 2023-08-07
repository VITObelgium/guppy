import random
import os
import time

import rasterio
import numpy as np

from guppy2.raster_calc_utils import create_raster, generate_raster_response, perform_operation, data_to_rgba, process_raster_with_function_in_chunks, \
    process_raster_list_with_function_in_chunks
from sqlalchemy.orm import Session
from starlette import status
from starlette.responses import Response
from guppy2.db import schemas as s, models as m
from osgeo import gdal
import datetime


def raster_calculation(db: Session, body: s.RasterCalculationBody):
    print('raster_calculation')
    t = time.time()
    base_path = '/content/tifs/generated'
    raster_name = f'generated_{datetime.datetime.now().strftime("%Y-%m-%d")}_{str(random.randint(0, 10000000))}.tif'
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
    process_raster_list_with_function_in_chunks(path_list, os.path.join(base_path, raster_name), path_list[0],
                                                function_to_apply=perform_operation, function_arguments={'layer_args': arguments_list, 'output_rgb': body.rgb},
                                                chunks=10, output_bands=4 if body.rgb else 1, dtype=np.uint8 if body.rgb else None, out_nodata=255 if body.rgb else None)
    build_overview_tiles = [2, 4, 8, 16, 32, 64]
    image = gdal.Open(os.path.join(base_path, raster_name), 1)  # 0 = read-only, 1 = read-write.
    gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
    image.BuildOverviews("NEAREST", build_overview_tiles)
    del image
    print('raster_calculation 200', time.time() - t)
    if body.geoserver:
        geoserver_layer = create_raster(raster_name)
        return Response(content=geoserver_layer, status_code=status.HTTP_201_CREATED)
    else:
        return generate_raster_response(os.path.join(base_path, raster_name))
