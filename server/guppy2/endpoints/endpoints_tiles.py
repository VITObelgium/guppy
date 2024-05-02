import logging
import os
import sqlite3
import tempfile
from functools import lru_cache
from typing import Optional

import geopandas as gpd
from fastapi import HTTPException, Response
from shapely.geometry import box
from sqlalchemy.orm import Session

from guppy2.db.models import TileStatistics
from guppy2.endpoints.endpoint_utils import validate_layer_and_get_file_path
from guppy2.endpoints.tile_utils import tile2lonlat, add_item_to_request_counter
from guppy2.error import create_error

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


def search_tile(layer_name: str, field_name: str, search_string: str, zoom_level: int, db: Session):
    """Searches for a tile in a given layer based on search criteria.

    Args:
        layer_name (str): The name of the layer.
        field_name (str): The field name to search for in the tile.
        search_string (str): The search string to match in the fieldName.
        zoom_level (int): The zoom level of the tile.
        db (Session): The session object used for database operations.

    Returns:
        Response: A Response object with the matching tile data in JSON format, or an error response.

    Raises:
        HTTPException: If there is an error in the tile search process.
    """
    mb_file = validate_layer_and_get_file_path(db, layer_name)
    try:
        df = read_cached_mbtile_as_df(mb_file, zoom_level)
        if df is not None and not df.empty:
            if field_name not in df.columns:
                create_error(code=400, message=f"Field {field_name} not found in the tile")
            df[field_name] = df[field_name].astype(str)
            df = df[df[field_name].str.contains(search_string, case=False, na=False)].copy()
        if not df.empty:
            return Response(df.head(10).to_json(), media_type="application/json")
        else:
            create_error(code=204, message="Tile not found")
    except Exception as e:
        create_error(code=404, message=str(e))


@lru_cache(maxsize=128)
def read_cached_mbtile_as_df(mb_file, zoomLevel):
    """
    Reads a mbtile file and returns its content as a GeoDataFrame.
    uses lru cache to store the GeoDataFrame for future use.

    Args:
        mb_file (str): Path to the mbtile file.
        zoomLevel (int): The desired zoom level to read from the mbtile file.

    Returns:
        GeoDataFrame: The mbtile content as a GeoDataFrame.

    """
    df = gpd.read_file(mb_file, engine="pyogrio", read_geometry=False, ZOOM_LEVEL=zoomLevel)
    return df
