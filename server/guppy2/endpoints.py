# coding: utf-8

import math
import os
import time

import numpy as np
import rasterio
from fastapi import status
from fastapi.responses import Response
from pyproj import Transformer
from rasterio.windows import from_bounds
from shapely.geometry import box
from sqlalchemy.orm import Session

from guppy2.db import models as m
from guppy2.db import schemas as s


def healthcheck(db: Session):
    db.execute('SELECT 1')
    return 'OK'


def get_stats_for_bbox(db: Session, layer_name: str, bbox_left: float, bbox_bottom: float, bbox_right: float, bbox_top: float, token_data: [str] = None):
    t = time.time()
    layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_name).first()
    if layer_model:
        path = layer_model.file_path
        if os.path.exists(path) and bbox_left and bbox_bottom and bbox_right and bbox_top:
            with rasterio.open(path) as src:
                transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
                bbox_bottom, bbox_left = transformer.transform(bbox_bottom, bbox_left)
                bbox_top, bbox_right = transformer.transform(bbox_top, bbox_right)
                res = max(src.res)
                pixels = (bbox_right - bbox_left) / res * (bbox_top - bbox_bottom) / res
                factor = int(pixels / 2000000)
                if factor < 1:
                    factor = 1
                bb_input = box(bbox_bottom, bbox_left, bbox_top, bbox_right)
                bb_raster = box(src.bounds[0], src.bounds[1], src.bounds[2], src.bounds[3])
                intersection = bb_input.intersection(bb_raster)
                if not intersection.is_empty:
                    window = from_bounds(intersection.bounds[0], intersection.bounds[1], intersection.bounds[2], intersection.bounds[3], src.transform).round_offsets()
                    rst = src.read(1,
                                   window=window,
                                   out_shape=(1, int(src.height / factor), int(src.width / factor))
                                   )
                    if rst.size != 0:
                        rst[rst == src.nodata] = np.nan
                        q2, q5, q95, q98 = np.nanquantile(rst, [0.02, 0.05, 0.95, 0.98])
                        response = s.StatsResponse(type='stats bbox',
                                                   min=float(np.nanmin(rst)),
                                                   max=float(np.nanmax(rst)),
                                                   sum=float(np.nansum(rst)),
                                                   mean=float(np.nanmean(rst)),
                                                   count=int(np.sum(np.isfinite(rst))),
                                                   q02=float(q2),
                                                   q05=float(q5),
                                                   q95=float(q95),
                                                   q98=float(q98),
                                                   )
                        print('get_stats_for_bbox', time.time() - t)
                        return response, 200
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return "result not found", 404


def get_point_value_from_raster(db: Session, layer_name: str, x: float, y: float, token_data: [str] = None):
    layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_name).first()
    if layer_model:
        path = layer_model.file_path
        with rasterio.open(path) as src:
            nodata = src.nodata
            transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
            x_, y_ = transformer.transform(x, y)
            for v in src.sample([(x_, y_)]):
                return s.PointResponse(type='point value', layer_name=layer_name, value=None if math.isclose(float(v[0]), nodata) else float(v[0]))
    return Response(status_code=status.HTTP_204_NO_CONTENT)