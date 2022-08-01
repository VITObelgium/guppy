# coding: utf-8

import math
import os
import time

import numpy as np
import rasterio
from fastapi import status
from fastapi.responses import Response
from joblib import Parallel, delayed
from pyproj import Transformer
from rasterio.windows import from_bounds
from shapely import wkt
from shapely.geometry import box
from shapely.ops import transform
from sqlalchemy.orm import Session

from guppy2.db import models as m
from guppy2.db import schemas as s
from guppy2.endpoint_utils import get_overview_factor, create_stats_response, _extract_area_from_dataset, _extract_shape_mask_from_dataset


def healthcheck(db: Session):
    db.execute('SELECT 1')
    return 'OK'


def get_stats_for_bbox(db: Session, layer_name: str, bbox_left: float, bbox_bottom: float, bbox_right: float, bbox_top: float, native: bool):
    t = time.time()
    layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_name).first()
    if layer_model:
        path = layer_model.file_path[1:]
        if os.path.exists(path) and bbox_left and bbox_bottom and bbox_right and bbox_top:
            transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
            bbox_bottom, bbox_left = transformer.transform(bbox_bottom, bbox_left)
            bbox_top, bbox_right = transformer.transform(bbox_top, bbox_right)
            overview_factor, overview_bin = get_overview_factor((bbox_bottom, bbox_left, bbox_top, bbox_right), native, path)
            with rasterio.open(path, overview_level=overview_factor) as src:
                bb_input = box(bbox_bottom, bbox_left, bbox_top, bbox_right)
                bb_raster = box(src.bounds[0], src.bounds[1], src.bounds[2], src.bounds[3])
                intersection = bb_input.intersection(bb_raster)
                if not intersection.is_empty:
                    window = from_bounds(intersection.bounds[0], intersection.bounds[1], intersection.bounds[2], intersection.bounds[3], src.transform).round_offsets()
                    rst = src.read(1, window=window, )
                    if rst.size != 0:
                        response = create_stats_response(rst, np.zeros_like(rst).astype(bool), src.nodata, f'bbox stats. Overview level: {overview_factor}, {overview_bin} scale')
                        print('get_stats_for_bbox 200', time.time() - t)
                        return response
        print('get_stats_for_bbox 204', time.time() - t)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    print('get_stats_for_bbox 404', time.time() - t)
    return Response(status_code=status.HTTP_404_NOT_FOUND)


def get_data_for_wkt(db: Session, layer_name: str, body: s.GeometryBody):
    t = time.time()
    layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_name).first()
    if layer_model:
        path = layer_model.file_path
        if os.path.exists(path) and body:
            geom = wkt.loads(body.geometry)
            if geom.is_valid:
                geom = transform(Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform, geom)
                if geom.is_valid:
                    with rasterio.open(path) as src:
                        if geom.area / (src.res[0] * src.res[1]) > 100000:
                            return Response(content=f'geometry area too large ({geom.area}m². allowed <={100000 * (src.res[0] * src.res[1])}m²)',
                                            status_code=status.HTTP_406_NOT_ACCEPTABLE)
                        rst, _ = _extract_area_from_dataset(src, geom, crop=True)
                        if rst.size != 0:
                            response = s.DataResponse(type='raw data', data=rst.tolist())
                            print('get_data_for_wkt 200', time.time() - t)
                            return response
        print('get_data_for_wkt 204', time.time() - t)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    print('get_data_for_wkt 404', time.time() - t)
    return Response(status_code=status.HTTP_404_NOT_FOUND)


def get_stats_for_wkt(db: Session, layer_name: str, body: s.GeometryBody, native: bool):
    t = time.time()
    layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_name).first()
    if layer_model:
        path = layer_model.file_path
        if os.path.exists(path) and body:
            geom = wkt.loads(body.geometry)
            if geom.is_valid:
                geom = transform(Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform, geom)
                if geom.is_valid:
                    overview_factor, overview_bin = get_overview_factor(geom.bounds, native, path)
                    with rasterio.open(path, overview_level=overview_factor) as src:
                        rst, rast_transform = _extract_area_from_dataset(src, geom, crop=True)
                        shape_mask = _extract_shape_mask_from_dataset(src, geom, crop=True)
                        if rst.size != 0:
                            response = create_stats_response(rst, shape_mask, src.nodata,
                                                             type=f'stats wkt. Overview level: {overview_factor}, {overview_bin} scale')
                            print('get_stats_for_wkt 200', time.time() - t)
                            return response
        print('get_stats_for_wkt 204', time.time() - t)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    print('get_stats_for_wkt 404', time.time() - t)
    return Response(status_code=status.HTTP_404_NOT_FOUND)


def get_line_data_for_wkt(db: Session, layer_name: str, body: s.LineGeometryBody):
    t = time.time()
    layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_name).first()
    if layer_model:
        path = layer_model.file_path
        if os.path.exists(path) and body:
            line = wkt.loads(body.geometry)
            if line.is_valid:
                line = transform(Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform, line)
                if line.is_valid:
                    distances = np.linspace(0, line.length, body.number_of_points)
                    points = [line.interpolate(distance) for distance in distances]
                    coords = [(point.x, point.y) for point in points]
                    if line.is_valid:
                        result = []
                        with rasterio.open(path) as src:
                            for v in src.sample(coords, indexes=1):
                                result.append(v[0])
                        response = s.LineDataResponse(type='line data', data=result)
                        print('get_data_for_wkt 200', time.time() - t)
                        return response
        print('get_data_for_wkt 204', time.time() - t)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    print('get_data_for_wkt 404', time.time() - t)
    return Response(status_code=status.HTTP_404_NOT_FOUND)


def get_line_data_list_for_wkt(db: Session, body: s.LineGeometryListBody):
    t = time.time()
    layer_models = db.query(m.LayerMetadata).filter(m.LayerMetadata.layer_name.in_(body.layer_names)).all()
    coords = None
    if layer_models:
        line = wkt.loads(body.geometry)
        if line.is_valid:
            line = transform(Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform, line)
            if line.is_valid:
                distances = np.linspace(0, line.length, body.number_of_points)
                points = [line.interpolate(distance) for distance in distances]
                coords = [(point.x, point.y) for point in points]
        if coords:
            result = Parallel(n_jobs=-1, prefer='threads')(delayed(sample_coordinates)(coords, layer_model) for layer_model in layer_models)
            if result:
                print('get_line_data_list_for_wkt 200', time.time() - t)
                return result
        print('get_line_data_list_for_wkt 204', time.time() - t)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    print('get_line_data_list_for_wkt 404', time.time() - t)
    return Response(status_code=status.HTTP_404_NOT_FOUND)


def sample_coordinates(coords, layer_model):
    result = []
    path = layer_model.file_path[1:]
    if os.path.exists(path):
        with rasterio.open(path) as src:
            for v in src.sample(coords, indexes=1):
                result.append(v[0])
    return s.LineData(layer_name=layer_model.layer_name, data=result)


def get_point_value_from_raster(db: Session, layer_name: str, x: float, y: float):
    t = time.time()
    layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_name).first()
    if layer_model:
        path = layer_model.file_path
        if os.path.exists(path) and x and y:
            with rasterio.open(path) as src:
                nodata = src.nodata
                transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
                x_, y_ = transformer.transform(x, y)
                for v in src.sample([(x_, y_)], indexes=1):
                    print('get_point_value_from_raster 200', time.time() - t)
                    return s.PointResponse(type='point value', layer_name=layer_name, value=None if math.isclose(float(v[0]), nodata) else float(v[0]))
    print('get_point_value_from_raster 204', time.time() - t)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def get_layer_mapping(db):
    t = time.time()
    layer_model = db.query(m.LayerMetadata).all()
    if layer_model:
        print('get_layer_mapping 200', time.time() - t)
        return layer_model
    print('get_layer_mapping 204', time.time() - t)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
