# coding: utf-8

import math
import os
import time

import geopandas as gpd
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

from guppy2.config import config as cfg
from guppy2.db import models as m
from guppy2.db import schemas as s
from guppy2.endpoint_utils import get_overview_factor, create_stats_response, _extract_area_from_dataset, _extract_shape_mask_from_dataset, _decode


def healthcheck(db: Session):
    db.execute('SELECT 1')
    return 'OK'


def get_stats_for_bbox(db: Session, layer_name: str, bbox_left: float, bbox_bottom: float, bbox_right: float, bbox_top: float, native: bool):
    t = time.time()
    layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_name).first()
    if layer_model:
        path = layer_model.file_path
        if os.path.exists(path) and bbox_left and bbox_bottom and bbox_right and bbox_top:
            with rasterio.open(path) as src:
                target_srs = src.crs.to_epsg()
            transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{target_srs}")
            bbox_bottom, bbox_left = transformer.transform(bbox_bottom, bbox_left)
            bbox_top, bbox_right = transformer.transform(bbox_top, bbox_right)
            overview_factor, overview_bin = get_overview_factor((bbox_bottom, bbox_left, bbox_top, bbox_right), native, path)
            with rasterio.open(path, overview_level=overview_factor) as src:
                bb_input = box(bbox_bottom, bbox_left, bbox_top, bbox_right)
                bb_raster = box(src.bounds[0], src.bounds[1], src.bounds[2], src.bounds[3])
                intersection = bb_input.intersection(bb_raster)
                if not intersection.is_empty:
                    window = from_bounds(intersection.bounds[0], intersection.bounds[1], intersection.bounds[2], intersection.bounds[3], src.transform).round_offsets()
                    if layer_model.is_rgb:
                        data = src.read(window=window, )
                        rst = _decode(data)
                    else:
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
            with rasterio.open(path) as src:
                target_srs = src.crs.to_epsg()
            if geom.is_valid:
                geom = transform(Transformer.from_crs(body.srs if body.srs else "EPSG:4326", f"EPSG:{target_srs}", always_xy=True).transform, geom)
                if geom.is_valid:
                    with rasterio.open(path) as src:
                        if geom.area / (src.res[0] * src.res[1]) > cfg.guppy.size_limit:
                            return Response(content=f'geometry area too large ({geom.area}m². allowed <={cfg.guppy.size_limit * (src.res[0] * src.res[1])}m²)',
                                            status_code=status.HTTP_406_NOT_ACCEPTABLE)
                        try:
                            rst, _ = _extract_area_from_dataset(src, [geom], crop=True, is_rgb=layer_model.is_rgb)
                            if layer_model.is_rgb:
                                rst = _decode(rst)
                        except ValueError as e:
                            return Response(content=str(e), status_code=status.HTTP_406_NOT_ACCEPTABLE)
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
            with rasterio.open(path) as src:
                target_srs = src.crs.to_epsg()
            if geom.is_valid:
                geom = transform(Transformer.from_crs(body.srs if body.srs else "EPSG:4326", f"EPSG:{target_srs}", always_xy=True).transform, geom)
                if geom.is_valid:
                    overview_factor, overview_bin = get_overview_factor(geom.bounds, native, path)
                    with rasterio.open(path, overview_level=overview_factor) as src:
                        try:
                            rst, _ = _extract_area_from_dataset(src, [geom], crop=True, is_rgb=layer_model.is_rgb)
                            if layer_model.is_rgb:
                                rst = _decode(rst)
                            shape_mask = _extract_shape_mask_from_dataset(src, [geom], crop=True)
                        except ValueError as e:
                            return Response(content=str(e), status_code=status.HTTP_406_NOT_ACCEPTABLE)
                    if rst.size != 0:
                        response = create_stats_response(rst, shape_mask, src.nodata,
                                                         type=f'stats wkt. Overview level: {overview_factor}, {overview_bin} scale')
                        print('get_stats_for_wkt 200', time.time() - t)
                        return response
        print('get_stats_for_wkt 204', time.time() - t)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    print('get_stats_for_wkt 404', time.time() - t)
    return Response(status_code=status.HTTP_404_NOT_FOUND)


def get_stats_for_model(layer_model, native, geom, srs):
    path = layer_model.file_path
    if os.path.exists(path) and geom:
        with rasterio.open(path) as src:
            target_srs = src.crs.to_epsg()
        if geom.is_valid:
            geom = transform(Transformer.from_crs(srs, f"EPSG:{target_srs}", always_xy=True).transform, geom)
            if geom.is_valid:
                overview_factor, overview_bin = get_overview_factor(geom.bounds, native, path)
                with rasterio.open(path, overview_level=overview_factor) as src:
                    try:
                        rst, _ = _extract_area_from_dataset(src, [geom], crop=True, is_rgb=layer_model.is_rgb)
                        if layer_model.is_rgb:
                            rst = _decode(rst)
                        shape_mask = _extract_shape_mask_from_dataset(src, [geom], crop=True)
                    except ValueError as e:
                        return Response(content=str(e), status_code=status.HTTP_406_NOT_ACCEPTABLE)
                if rst.size != 0:
                    return create_stats_response(rst, shape_mask, src.nodata,
                                                 type=f'stats wkt. Overview level: {overview_factor}, {overview_bin} scale',
                                                 layer_name=layer_model.layer_name
                                                 )
    return None


def get_stats_for_wkt_list(db: Session, body: s.GeometryBodyList, native: bool):
    t = time.time()
    layer_models = db.query(m.LayerMetadata).filter(m.LayerMetadata.layer_name.in_(body.layer_names)).all()
    if layer_models:
        geom = wkt.loads(body.geometry)
        srs = body.srs if body.srs else "EPSG:4326"
        result = Parallel(n_jobs=-1, prefer='threads')(delayed(get_stats_for_model)(layer_model, native, geom, srs) for layer_model in layer_models)
        if result:
            print('get_stats_for_wkt 200', time.time() - t)
            return result
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
            with rasterio.open(path) as src:
                target_srs = src.crs.to_epsg()
            if line.is_valid:
                line = transform(Transformer.from_crs(body.srs if body.srs else "EPSG:4326", f"EPSG:{target_srs}", always_xy=True).transform, line)
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
            # result = Parallel(n_jobs=4, prefer='threads')(delayed(sample_coordinates)(coords, layer_model.file_path[1:], layer_model.layer_name) for layer_model in layer_models)
            print('get_line_data_list_for_wkt pre sample', time.time() - t)
            result = sample_coordinates_window(coords, layer_models, line.bounds)
            if result:
                print('get_line_data_list_for_wkt 200', time.time() - t)
                return result
        print('get_line_data_list_for_wkt 204', time.time() - t)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    print('get_line_data_list_for_wkt 404', time.time() - t)
    return Response(status_code=status.HTTP_404_NOT_FOUND)


def sample_coordinates(coords, path, layer_name):
    result = []
    if os.path.exists(path):
        with rasterio.open(path) as src:
            x = src.sample(coords, indexes=1)
            for v in x:
                result.append(v[0])
    else:
        print("sample_coordinates file not found ", path)
    return s.LineData(layer_name=layer_name, data=result)


def sample_coordinates_window(coords, layer_models, bounds):
    result_all = []
    path = layer_models[0].file_path
    with rasterio.open(path) as src:
        window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], src.transform).round_offsets()
        rows, cols = src.index([p[0] for p in coords], [p[1] for p in coords])
        in_rows = []
        in_cols = []
        out_idx = []
        in_idx = []
        data = src.read(1, window=window)
        r_max, c_max = data.shape
        for i, (r, c) in enumerate(zip(rows, cols)):
            if r < 0 or c < 0 or r > r_max or c > c_max:
                out_idx.append(i)
            else:
                in_idx.append(i)
                in_rows.append(r)
                in_cols.append(c)

    # result_all = Parallel(n_jobs=4, prefer='processes')(delayed(sample_layer)(in_cols, in_idx, in_rows, layer_model, out_idx, window) for layer_model in layer_models)
    for layer_model in layer_models:
        result_all.append(sample_layer(in_cols, in_idx, in_rows, layer_model, out_idx, window))
    return result_all


def sample_layer(in_cols, in_idx, in_rows, layer_model, out_idx, window):
    path = layer_model.file_path
    with rasterio.open(path) as src:
        data = src.read(1, window=window)
    result = {}
    f = data[in_rows, in_cols]
    for i, v in zip(in_idx, f):
        result[i] = v
    for i in out_idx:
        result[i] = -9999
    result = [result[key] for key in sorted(result.keys())]
    return s.LineData(layer_name=layer_model.layer_name, data=result)


def get_point_value_from_raster(db: Session, layer_name: str, x: float, y: float):
    t = time.time()
    layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_name).first()
    if layer_model:
        path = layer_model.file_path
        if os.path.exists(path) and x and y:
            with rasterio.open(path) as src:
                nodata = src.nodata
                target_srs = src.crs.to_epsg()
                transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{target_srs}", always_xy=True)
                x_, y_ = transformer.transform(x, y)
                for v in src.sample([(x_, y_)], indexes=1):
                    print('get_point_value_from_raster 200', time.time() - t)
                    return s.PointResponse(type='point value', layer_name=layer_name, value=None if math.isclose(float(v[0]), nodata) else float(v[0]))
    print('get_point_value_from_raster 204', time.time() - t)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def get_layer_mapping(db, layer_name):
    t = time.time()
    layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_name).first()
    if layer_model:
        print('get_layer_mapping 200', time.time() - t)
        return layer_model
    print('get_layer_mapping 204', time.time() - t)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def get_layers_mapping(db, limit=100, offset=0):
    t = time.time()
    layer_model = db.query(m.LayerMetadata).limit(limit).offset(offset).all()
    if layer_model:
        print('get_layers_mapping 200', time.time() - t)
        return layer_model
    print('get_layers_mapping 204', time.time() - t)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def get_line_object_list_for_wkt(db: Session, layer_name: str, body: s.LineObjectGeometryBody):
    t = time.time()
    layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_name).first()
    line_points_df = None
    if layer_model:
        line = wkt.loads(body.geometry)
        if line.is_valid:
            line = transform(Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform, line)
            if line.is_valid:
                distances = np.linspace(0, line.length, body.number_of_points)
                points = [line.interpolate(distance) for distance in distances]
                line_points_df = gpd.GeoDataFrame(crs='epsg:3857', geometry=points)
        if line_points_df is not None:
            input_file_df = gpd.read_file(layer_model.file_path)
            input_file_df.to_crs(crs='epsg:3857', inplace=True)
            result_df = gpd.sjoin_nearest(input_file_df, line_points_df, max_distance=int(body.distance), distance_col='join_dist')
            result_df = result_df.loc[result_df.groupby(['index_right', 'modeleenheid'])['join_dist'].idxmin()]  # keep closest
            result_df.fillna('', inplace=True)
            result = result_df.to_dict(orient='records')
            if result:
                print('get_line_object_list_for_wkt 200', time.time() - t)
                return result
        print('get_line_object_list_for_wkt 204', time.time() - t)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    print('get_line_object_list_for_wkt 404', time.time() - t)
    return Response(status_code=status.HTTP_404_NOT_FOUND)


def get_classification_for_wkt(db: Session, layer_name: str, body: s.GeometryBody):
    t = time.time()
    layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_name).first()
    if layer_model:
        path = layer_model.file_path
        if os.path.exists(path) and body:
            geom = wkt.loads(body.geometry)
            with rasterio.open(path) as src:
                target_srs = src.crs.to_epsg()
            if geom.is_valid:
                geom = transform(Transformer.from_crs(body.srs if body.srs else "EPSG:4326", f"EPSG:{target_srs}", always_xy=True).transform, geom)
                if geom.is_valid:
                    with rasterio.open(path) as src:
                        if geom.area / (src.res[0] * src.res[1]) > cfg.guppy.size_limit:
                            return Response(content=f'geometry area too large ({geom.area}m². allowed <={cfg.guppy.size_limit * (src.res[0] * src.res[1])}m²)',
                                            status_code=status.HTTP_406_NOT_ACCEPTABLE)
                        try:
                            rst, _ = _extract_area_from_dataset(src, [geom], crop=True, is_rgb=layer_model.is_rgb)
                            if layer_model.is_rgb:
                                rst = _decode(rst)
                            shape_mask = _extract_shape_mask_from_dataset(src, [geom], crop=True)
                        except ValueError as e:
                            return Response(content=str(e), status_code=status.HTTP_406_NOT_ACCEPTABLE)
                    if rst.size != 0:
                        values, counts = np.unique(np.where(shape_mask == 0, rst, -999999999999), return_counts=True)
                        result_classes = []
                        total_count = sum([c for v, c in zip(values, counts) if v != -999999999999])
                        for v, c in zip(values, counts):
                            if v != -999999999999:
                                result_classes.append(s.ClassificationEntry(value=v, count=c, percentage=c / total_count * 100))
                        response = s.ClassificationResult(type='classification', data=result_classes)
                        print('classification_for_wkt 200', time.time() - t)
                        return response
            print('classification_for_wkt invalid geometry', time.time() - t)
            return Response(content="invalid geometry", status_code=status.HTTP_406_NOT_ACCEPTABLE)
    print('classification_for_wkt 204', time.time() - t)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def get_combine_layers(db: Session, body: s.CombineLayersGeometryBody):
    t = time.time()
    for layer_item in body.layer_list:
        geom = wkt.loads(body.geometry)
        layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_item.layer_name).first()
        if layer_model:
            path = layer_model.file_path
            if os.path.exists(path) and body:
                with rasterio.open(path) as src:
                    target_srs = src.crs.to_epsg()
                if geom.is_valid:
                    geom = transform(Transformer.from_crs("EPSG:4326", f"EPSG:{target_srs}", always_xy=True).transform, geom)
                    if geom.is_valid:
                        with rasterio.open(path) as src:
                            if geom.area / (src.res[0] * src.res[1]) > cfg.guppy.size_limit:
                                return Response(content=f'geometry area too large ({geom.area}m². allowed <={cfg.guppy.size_limit * (src.res[0] * src.res[1])}m²)',
                                                status_code=status.HTTP_406_NOT_ACCEPTABLE)
                            try:
                                rst, _ = _extract_area_from_dataset(src, [geom], crop=True, is_rgb=layer_model.is_rgb)
                                if layer_model.is_rgb:
                                    rst = _decode(rst)
                            except ValueError as e:
                                return Response(content=str(e), status_code=status.HTTP_406_NOT_ACCEPTABLE)
                        if rst.size != 0:
                            response = s.DataResponse(type='raw data', data=rst.tolist())
                            print('get_combine_layers 200', time.time() - t)
                            return response
            print('get_combine_layers 204', time.time() - t)
            return Response(status_code=status.HTTP_204_NO_CONTENT)
    print('get_combine_layers 404', time.time() - t)
    return Response(status_code=status.HTTP_404_NOT_FOUND)
