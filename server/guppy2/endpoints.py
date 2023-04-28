# coding: utf-8

import math
import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from fastapi import status
from fastapi.responses import Response, ORJSONResponse
from joblib import Parallel, delayed
from pyproj import Transformer
from rasterio.windows import from_bounds
from rasterio.features import dataset_features
from shapely import wkt
from shapely.geometry import box, MultiLineString
from shapely.ops import transform
from sqlalchemy.orm import Session
from sqlalchemy import text

from guppy2.config import config as cfg
from guppy2.db import models as m
from guppy2.db import schemas as s
from guppy2.endpoint_utils import get_overview_factor, create_stats_response, _extract_area_from_dataset, _extract_shape_mask_from_dataset, _decode


def healthcheck(db: Session):
    db.execute(text('SELECT 1'))
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
    coords = {}
    if layer_models:
        line = wkt.loads(body.geometry)
        if line.is_valid:
            line = transform(Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform, line)
            if line.is_valid:
                distances = np.linspace(0, line.length, body.number_of_points)
                points = [line.interpolate(distance) for distance in distances]
                coords['1'] = [(point.x, point.y) for point in points]
        if coords:
            print('get_line_data_list_for_wkt pre sample', time.time() - t)
            result = sample_coordinates_window(coords, layer_models, line.bounds)
            if result:
                print('get_line_data_list_for_wkt 200', time.time() - t)
                return ORJSONResponse(content=result)
        print('get_line_data_list_for_wkt 204', time.time() - t)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    print('get_line_data_list_for_wkt 404', time.time() - t)
    return Response(status_code=status.HTTP_404_NOT_FOUND)


def get_multi_line_data_list_for_wkt(db: Session, body: s.MultiLineGeometryListBody):
    t = time.time()
    layer_models = db.query(m.LayerMetadata).filter(m.LayerMetadata.layer_name.in_(body.layer_names)).all()
    coords_list = {}
    result_per_line = []
    epsg_lines = []
    if layer_models:
        lines = [wkt.loads(line) for line in body.geometry]
        for line in lines:
            if line.is_valid:
                line = transform(Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform, line)
                if line.is_valid:
                    distances = np.linspace(0, line.length, body.number_of_points)
                    points = [line.interpolate(distance) for distance in distances]
                    coords_list[line] = [(point.x, point.y) for point in points]
                    epsg_lines.append(line)
        if coords_list:
            print('get_multi_line_data_list_for_wkt pre sample', time.time() - t)
            result = sample_coordinates_window(coords_list, layer_models, MultiLineString(epsg_lines).bounds, body.round_val)
            start_data = 0
            end_data = body.number_of_points
            for line in body.geometry:
                datalist = []
                for r in result:
                    datalist.append({'layerName': r['layerName'], 'data': r['data'][start_data:end_data]})
                result_per_line.append({'key': line, 'lineData': datalist})
                start_data += body.number_of_points
                end_data += body.number_of_points

            if result_per_line:
                print('get_multi_line_data_list_for_wkt 200', time.time() - t)
                return ORJSONResponse(content=result_per_line)
        print('get_multi_line_data_list_for_wkt 204', time.time() - t)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    print('get_multi_line_data_list_for_wkt 404', time.time() - t)
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


def sample_coordinates_window(coords_dict, layer_models, bounds, round_val=None):
    result_all = []
    path = layer_models[0].file_path
    coords = []
    for k, v in coords_dict.items():
        coords.extend(v)
    with rasterio.open(path) as src:
        geometry_window = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], src.transform).round_offsets()
        rows, cols = src.index([p[0] for p in coords], [p[1] for p in coords])
        cols = [c - geometry_window.col_off for c in cols]
        rows = [r - geometry_window.row_off for r in rows]
        in_rows = []
        in_cols = []
        out_idx = []
        in_idx = []
        data = src.read(1, window=geometry_window)
        clipped_data = np.full((max(int(geometry_window.height), data.shape[0]), max(data.shape[1], int(geometry_window.width))), fill_value=0, dtype=np.float32)
        if data.shape == clipped_data.shape:
            clipped_data = data
        else:
            transform = rasterio.windows.transform(geometry_window, src.transform)
            # Clip the raster using the geometry
            # Update the portion of clipped_data that overlaps with the geometry
            window = rasterio.windows.from_bounds(*bounds, transform=transform)
            row_offset = abs(int(geometry_window.row_off)) if geometry_window.row_off < 0 else 0
            col_offset = abs(int(geometry_window.col_off)) if geometry_window.col_off < 0 else 0
            clipped_data[row_offset:row_offset + int(data.shape[0]), col_offset:col_offset + int(data.shape[1])] = data

        # create the metadata for the dataset

        r_max, c_max = clipped_data.shape
        for i, (r, c) in enumerate(zip(rows, cols)):
            if r < 0 or c < 0 or r >= r_max or c >= c_max:
                out_idx.append(i)
            else:
                in_idx.append(i)
                in_rows.append(r)
                in_cols.append(c)

    for layer_model in layer_models:
        result_all.append(sample_layer(in_cols, in_idx, in_rows, layer_model, out_idx, geometry_window, round_val, bounds))
    return result_all


def sample_layer(in_cols, in_idx, in_rows, layer_model, out_idx, geometry_window, round_val: int = None, bounds=None):
    path = layer_model.file_path
    with rasterio.open(path) as src:
        data = src.read(1, window=geometry_window)
        nodata = src.nodata
        if nodata is None:
            nodata = -9999

        clipped_data = np.full((max(int(geometry_window.height), data.shape[0]), max(data.shape[1], int(geometry_window.width))), fill_value=nodata, dtype=np.float32)
        if data.shape == clipped_data.shape:
            clipped_data = data
        else:
            transform = rasterio.windows.transform(geometry_window, src.transform)
            # Clip the raster using the geometry
            # Update the portion of clipped_data that overlaps with the geometry
            window = rasterio.windows.from_bounds(*bounds, transform=transform)
            row_offset = abs(int(geometry_window.row_off)) if geometry_window.row_off < 0 else 0
            col_offset = abs(int(geometry_window.col_off)) if geometry_window.col_off < 0 else 0
            clipped_data[row_offset:row_offset + int(data.shape[0]), col_offset:col_offset + int(data.shape[1])] = data

    result = {}
    if round_val:
        clipped_data = np.round(clipped_data, round_val)
    f = clipped_data[in_rows, in_cols]
    for i, v in zip(in_idx, f):
        result[i] = v
    for i in out_idx:
        result[i] = nodata
    result = [result[key] for key in sorted(result.keys())]
    return {'layerName': layer_model.layer_name, 'data': result}


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
        if body.geometry:
            line = wkt.loads(body.geometry)
            if line.is_valid:
                line = transform(Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform, line)
                if line.is_valid:
                    distances = np.linspace(0, line.length, body.number_of_points)
                    points = [line.interpolate(distance) for distance in distances]
                    line_points_df = gpd.GeoDataFrame(crs='epsg:3857', geometry=points)
        if layer_model.file_path.endswith('.pkl'):
            input_file_df = pd.read_pickle(layer_model.file_path)
        else:
            input_file_df = gpd.read_file(layer_model.file_path)
        input_file_df.to_crs(crs='epsg:3857', inplace=True)
        input_file_df = input_file_df[~pd.isna(input_file_df.geometry)]
        input_file_df = input_file_df[~input_file_df.is_empty]
        if line_points_df is not None:
            result_df = gpd.sjoin_nearest(input_file_df, line_points_df, max_distance=int(body.distance), distance_col='join_dist')
            result_df = result_df.loc[result_df.groupby(['index_right', 'modeleenheid'])['join_dist'].idxmin()]  # keep closest
            result_df.fillna('', inplace=True)
            result_df = result_df.to_wkt()
            result_df = pd.DataFrame(result_df)
            result = result_df.to_dict(orient='records')
        else:
            input_file_df.fillna('', inplace=True)
            input_file_df = input_file_df.to_wkt()
            result_df = pd.DataFrame(input_file_df)
            result = result_df.to_dict(orient='records')
        if result:
            print('get_line_object_list_for_wkt 200', time.time() - t)
            return ORJSONResponse(result)
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


def get_layer_contour(layer):
    with rasterio.open(layer.file_path) as src:
        contour_geojson = list(dataset_features(src, bidx=1, as_mask=True, precision=1, band=False, geographic=False))
        return {'layerName': layer.layer_name, 'geometry': contour_geojson}


def get_countour_for_models(db: Session, body: s.CountourBodyList):
    t = time.time()
    layer_models = db.query(m.LayerMetadata).filter(m.LayerMetadata.layer_name.in_(body.layer_names)).all()
    if layer_models:
        result = Parallel(n_jobs=-1, prefer='threads')(delayed(get_layer_contour)(layer) for layer in layer_models)
        if result:
            print('get_countour_for_models 200', time.time() - t)
            return ORJSONResponse(content=result)
        print('get_countour_for_models 204', time.time() - t)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    print('get_countour_for_models 404', time.time() - t)
    return Response(status_code=status.HTTP_404_NOT_FOUND)
