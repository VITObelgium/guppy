import logging
import os
import sqlite3
from functools import lru_cache
from typing import Optional

from fastapi import HTTPException, Response
from sqlalchemy.orm import Session

from guppy2.db.models import LayerMetadata

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
        with sqlite3.connect(mb_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?", (z, x, y))
            tile = cursor.fetchone()
            if tile:
                return bytes(tile[0])
            else:
                return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def log_cache_info():
    """
    Logs the cache hits and cache misses information.

    Raises:
        None

    Returns:
        None
    """
    cache_info = get_tile_data.cache_info()
    logger.info(f"Cache hits: {cache_info.hits}, Cache misses: {cache_info.misses}")


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
    layer = db.query(LayerMetadata).filter_by(layer_name=layer_name).first()
    if not layer:
        raise HTTPException(status_code=404, detail=f"Layer not found: {layer_name}")
    mb_file = layer.file_path
    if not os.path.exists(mb_file):
        raise HTTPException(status_code=404, detail=f"MBTiles file file not found: {mb_file}")

    try:
        tile_data = get_tile_data(layer_name, mb_file, z, x, y)
        log_cache_info()
        if tile_data:
            return Response(tile_data, media_type="application/x-protobuf", headers={"Content-Encoding": "gzip"})
        else:
            raise HTTPException(status_code=404, detail="Tile not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
