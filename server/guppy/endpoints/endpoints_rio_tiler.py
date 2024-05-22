"""rio-tiler tile server."""

import logging
import time
from functools import lru_cache

from fastapi import HTTPException
from osgeo import gdal
from rio_tiler.colormap import cmap, InvalidColorMapName
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import Reader
from rio_tiler.profiles import img_profiles
from sqlalchemy.orm import Session
from starlette.responses import Response

from guppy.endpoints.endpoint_utils import validate_layer_and_get_file_path
from guppy.endpoints.tile_utils import data_to_rgba

logger = logging.getLogger(__name__)


def get_tile_for_layer(layer_name: str, style: str, db: Session, z: int, x: int, y: int) -> Response:
    """
    Args:
        layer_name: A string representing the name of the layer.
        style: A string representing the style of the tile.
        db: A Session object representing the database session.
        z: An integer representing the zoom level of the tile.
        x: An integer representing the x coordinate of the tile.
        y: An integer representing the y coordinate of the tile.

    Returns:
        The tile for the given layer, style, zoom level, and coordinates.

    """
    t = time.time()
    file_path = validate_layer_and_get_file_path(db, layer_name)

    result = get_tile(file_path, z, x, y, style)
    log_cache_info(t)
    return result


def log_cache_info(t):
    """
    Logs the cache hits and cache misses information.

    Raises:
        None

    Returns:
        None
    """
    cache_info = get_tile.cache_info()
    logger.info(f"Cache hits: {cache_info.hits}, Cache misses: {cache_info.misses}, Time: {time.time() - t}")


@lru_cache(maxsize=128)
def get_cached_colormap(name):
    """
    Function to enable lru caching of colormaps.
    Args:
        name: The name of the colormap to retrieve.

    Returns:
        The cached colormap associated with the specified name.

    """
    return cmap.get(name)


@lru_cache(maxsize=128)
def get_tile(file_path: str, z: int, x: int, y: int, style: str = None) -> Response:
    """
    Args:
        file_path: A string representing the path to the file.
        z: An integer representing the zoom level.
        x: An integer representing the x-coordinate of the tile.
        y: An integer representing the y-coordinate of the tile.
        style: An optional string representing the style of the tile.

    Returns:
        A Response object containing the rendered tile image in PNG format, or raises an HTTPException with a status code of 404 and a corresponding detail message.

    """
    t = time.time()
    try:
        img = None
        nodata = None
        with Reader(file_path) as cog:
            try:
                img = cog.tile(x, y, z)
                nodata = cog.dataset.nodata
            except TileOutsideBounds:
                raise HTTPException(status_code=204, detail=f"Tile out of bounds {z} {x} {y}")
            if img.dataset_statistics is None:
                stats = cog.statistics()['b1']
                # generate statistics file for next time
                gdal.Info(file_path, computeMinMax=True, stats=True)
        if img:
            colormap = None
            add_mask = True
            if style:
                if style != 'shader_rgba':
                    try:
                        colormap = get_cached_colormap(style)
                        if img.dataset_statistics:
                            img.rescale(in_range=img.dataset_statistics)
                        else:
                            img.rescale(in_range=[(stats.min, stats.max)])
                    except InvalidColorMapName:
                        raise HTTPException(status_code=404, detail=f"Invalid colormap name: {style}")
                else:
                    img.array = data_to_rgba(img.data[0], nodata)
                    add_mask = False
            elif img.dataset_statistics:
                img.rescale(in_range=img.dataset_statistics)
            else:
                img.rescale(in_range=[(stats.min, stats.max)])
            content = img.render(img_format="PNG", colormap=colormap, add_mask=add_mask, **img_profiles.get("png"))
            return Response(content, media_type="image/png")
    except TileOutsideBounds:
        raise HTTPException(status_code=204, detail=f"Tile out of bounds {z} {x} {y}")
