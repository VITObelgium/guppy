import logging
import os
import sqlite3
import tempfile
import time
from functools import lru_cache
from typing import Optional

import geopandas as gpd
from fastapi import HTTPException, Response, Request
from fastapi.responses import StreamingResponse
from pygeofilter.backends.sql import to_sql_where
from pygeofilter.parsers.ecql import parse
from shapely.geometry import box
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from guppy.db.models import TileStatistics
from guppy.db.schemas import QueryParams
from guppy.endpoints.endpoint_utils import validate_layer_and_get_file_path
from guppy.endpoints.tile_utils import tile2lonlat, add_item_to_request_counter, get_field_mapping, FUNCTION_MAP
from guppy.error import create_error

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def get_tile_data(layer_name: str, mb_file: str, z: int, x: int, y: int) -> Optional[bytes]:
    """
    Args:
        layer_name: The name of the layer for which the tile data is being retrieved.
        mb_file: The path to the MBTiles file from which the tile data is being retrieved.
        z: The zoom level of the tile.
        x: The X coordinate of the tile.
        y: The Y coordinate of the tile.

    Returns:
        Optional[bytes]: The tile data as bytes if found, or None if no tile data exists for the given parameters.

    Raises:
        HTTPException: If there is an error retrieving the tile data from the MBTiles file.

    """

    # Flip Y coordinate because MBTiles grid is TMS (bottom-left origin)
    y = (1 << z) - 1 - y
    logger.info(f"Getting tile for layer {layer_name} at zoom {z}, x {x}, y {y}")
    try:
        uri = f'file:{mb_file}?mode=ro'
        with sqlite3.connect(uri, uri=True) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?", (z, x, y))
            tile = cursor.fetchone()
            if tile:
                return bytes(tile[0])
            else:
                return None
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


def log_cache_info():
    """
    Logs the cache hits and cache misses information.

    """
    cache_info = get_tile_data.cache_info()
    logger.info(f"Cache hits: {cache_info.hits}, Cache misses: {cache_info.misses}")


def clear_tile_cache():
    """
    Clears the tile cache.

    """
    get_tile_data.cache_clear()
    logger.info("Tile cache cleared")


def get_tile(layer_name: str, db: Session, z: int, x: int, y: int):
    """
    Args:
        layer_name (str): The name of the layer to retrieve the tile from.
        db (Session): The database session object.
        z (int): The zoom level of the tile.
        x (int): The x-coordinate of the tile.
        y (int): The y-coordinate of the tile.

    Raises:
        HTTPException: If the layer or MBTiles file is not found, or if an internal server error occurs.

    Returns:
        Response: The tile data as a Response object. The media type is set to "application/x-protobuf" and the content encoding is set to "gzip".
    """
    mb_file = validate_layer_and_get_file_path(db, layer_name)
    tile_data = None
    try:
        add_item_to_request_counter(layer_name, z, x, y)
        tile_data = get_tile_data(layer_name, mb_file, z, x, y)
        log_cache_info()
    except Exception as e:
        create_error(code=404, message=str(e))
    if tile_data:
        return Response(tile_data, media_type="application/x-protobuf", headers={"Content-Encoding": "gzip"})
    else:
        create_error(code=204, message="Tile not found")


def get_tile_statistics(db: Session, layer_name: str, offset: int = 0, limit: int = 20):
    """
    Retrieves the statistics for the specified layer.
    Args:
        db: The database session to query the statistics from.
        layer_name: The name of the layer to retrieve the statistics for.
        offset: The number of statistics to skip from the beginning of the result set. Defaults to 0.
        limit: The maximum number of statistics to retrieve. Defaults to 20.

    Returns:
        A list of TileStatistics objects representing the statistics for the specified layer.

    Raises:
        HTTPException: If an error occurs while querying the database.
    """
    try:
        stats = db.query(TileStatistics).filter_by(layer_name=layer_name).order_by(TileStatistics.count.desc()).offset(offset).limit(limit).all()
        return stats
    except Exception as e:
        create_error(code=404, message=str(e))


def get_tile_statistics_images(db: Session, layer_name: str):
    """
    Generates a GeoPackage file containing the tile statistics for the specified layer.
    Args:
        db: The database session object.
        layer_name: The name of the layer.

    Returns:
        The GeoPackage file contents as bytes.

    """
    tiles_info = db.query(TileStatistics).filter_by(layer_name=layer_name).all()
    min_z = min([tile.z for tile in tiles_info])
    max_z = max([tile.z for tile in tiles_info])
    temp_filepath = tempfile.mktemp(suffix='.gpkg')
    try:
        for z in range(min_z, max_z + 1):
            counts, polygons = [], []
            for tile in tiles_info:
                if tile.z != z:
                    continue
                flipped_y = (2 ** z - 1) - tile.y
                bounds = tile2lonlat(tile.x, flipped_y, z)
                polygon = box(*bounds)
                polygons.append(polygon)
                counts.append(tile.count)

            gdf = gpd.GeoDataFrame(data={'count': counts, 'z': z, 'geometry': polygons}, crs='EPSG:4326')

            gdf.to_file(temp_filepath, layer=f'z_{z}', driver='GPKG', append=True)
        with open(temp_filepath, 'rb') as f:
            gpkg_bytes = f.read()
        return gpkg_bytes
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)


def search_tile(layer_name: str, params: QueryParams, limit: int, offset: int, db: Session):
    mb_file = validate_layer_and_get_file_path(db, layer_name)
    filters = parse(params.cql_filter)
    sqlite_file = mb_file.replace(".mbtiles", ".sqlite")
    if not os.path.exists(mb_file):
        create_error(code=404, message=f"Sqlite file not found for layer {layer_name}")
    try:
        uri = f'file:{sqlite_file}?mode=ro'
        with sqlite3.connect(uri, uri=True) as conn:
            cursor = conn.cursor()
            field_mapping = get_field_mapping(conn)
            where = to_sql_where(filters, field_mapping, FUNCTION_MAP)
            cursor.execute(f"SELECT * FROM tiles WHERE {where} LIMIT ? OFFSET ?", (limit, offset))
            data = cursor.fetchall()
            # Fetch all rows as a list of dicts
            columns = [description[0] for description in cursor.description]
            rows = [dict(zip(columns, row)) for row in data]
            if rows:
                return rows
            create_error(code=204, message="No data found for the specified query.")
    except SQLAlchemyError as e:
        create_error(code=404, message=str(e))


def get_cog_result(layer_name: str, request: Request, db: Session):
    """
    Serve COG (Cloud Optimized GeoTIFF) files with HTTP range request support.
    This endpoint supports partial content requests which are essential for COG files.
    """
    t = time.time()
    file_path = validate_layer_and_get_file_path(db, layer_name, file_type=".tif")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    file_size = os.path.getsize(file_path)

    range_header = request.headers.get("range")

    if range_header:
        try:
            range_match = range_header.replace("bytes=", "").split("-")
            start = int(range_match[0]) if range_match[0] else 0
            end = int(range_match[1]) if range_match[1] else file_size - 1

            if start >= file_size or end >= file_size or start > end:
                raise HTTPException(status_code=416, detail="Range not satisfiable")

        except (ValueError, IndexError):
            raise HTTPException(status_code=400, detail="Invalid range header")

        def iterfile():
            with open(file_path, "rb") as file:
                file.seek(start)
                remaining = end - start + 1
                chunk_size = 8192
                while remaining > 0:
                    chunk = file.read(min(chunk_size, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
            "Content-Type": "image/tiff",
            "Cache-Control": "public, max-age=31536000",
            "ETag": str(hash({start}-{end})),
        }
        logger.info(f"Serving COG file {file_path}    {time.time() - t}")
        return StreamingResponse(iterfile(), status_code=206, headers=headers)

    else:
        # Return full file
        def iterfile():
            with open(file_path, "rb") as file:
                while chunk := file.read(8192):
                    yield chunk

        headers = {
            'Accept-Ranges': 'bytes',
            'Content-Length': str(file_size),
            'Content-Type': 'image/tiff'
        }
        logger.info(f"Serving full COG file {file_path}    {time.time() - t}")
        return StreamingResponse(iterfile(), status_code=200, headers=headers)
