import sqlite3
import logging
from fastapi import HTTPException, Response
from sqlalchemy.orm import Session

from guppy2.db.models import LayerMetadata

logger = logging.getLogger(__name__)

def get_tile(layer_name: str, db: Session, z: int, x: int, y: int):
    """
    Args:
        layer_name: The name of the layer to get the tile from.
        db: The database session to query from.
        z: The zoom level of the tile.
        x: The x-coordinate of the tile.
        y: The y-coordinate of the tile.

    Returns:
        If the tile exists in the database, it will return a Response object containing the tile data as image/png. If the tile does not exist, it will raise an HTTPException with status
    * code 404 and detail message "Tile not found". If there is an error accessing the database, it will raise an HTTPException with status code 500 and the error message as the detail.

    Raises:
        HTTPException: If the layer is not found in the database (status code 404), or if there is an error accessing the database (status code 500).
    """
    # Flip Y coordinate because MBTiles grid is TMS (bottom-left origin)
    y = (1 << z) - 1 - y
    logger.info(f"Getting tile for layer {layer_name} at zoom {z}, x {x}, y {y}")
    layer = db.query(LayerMetadata).filter_by(layer_name=layer_name).first()
    if not layer:
        raise HTTPException(status_code=404, detail="Layer not found")
    mb_file = layer.file_path
    try:
        with sqlite3.connect(mb_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?", (z, x, y))
            tile = cursor.fetchone()
            if tile:
                return Response(bytes(tile[0]), media_type="application/x-protobuf", headers={"Content-Encoding": "gzip"})
            else:
                raise HTTPException(status_code=404, detail="Tile not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
