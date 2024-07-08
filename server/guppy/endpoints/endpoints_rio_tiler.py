"""rio-tiler tile server."""

import logging
import time
from functools import lru_cache

import numpy as np
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


def get_tile_for_layer(layer_name: str, style: str, db: Session, z: int, x: int, y: int, values: str = None, colors: str = None) -> Response:
    """
    Args:
        layer_name: A string representing the name of the layer.
        style: A string representing the style of the tile.
        db: A Session object representing the database session.
        z: An integer representing the zoom level of the tile.
        x: An integer representing the x coordinate of the tile.
        y: An integer representing the y coordinate of the tile.
        values: An optional string representing the values for the custom style.
        colors: An optional string representing the colors for the custom style.

    Returns:
        The tile for the given layer, style, zoom level, and coordinates.

    """
    t = time.time()
    file_path = validate_layer_and_get_file_path(db, layer_name)

    result = get_tile(file_path, z, x, y, style, values, colors)
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


def is_hex_color(input):
    """
    Function to check if a string is a valid hex color.
    Args:
        input: The string to check.

    Returns:
        True if the input is a valid hex color, False otherwise.

    """
    return len(input) == 6 and all(c in "0123456789ABCDEF" for c in input.upper())


def hex_to_rgb(hex_color):
    """
    Function to convert a hex color to an RGB color.
    Args:
        hex_color: The hex color to convert.

    Returns:
        A tuple containing the RGB color.

    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def generate_colormap(min_val, max_val, value_points, colors):
    # Map the provided value_points from [min_val, max_val] to [0, 255]
    rescaled_points = np.interp(value_points, (min_val, max_val), (0, 255))
    # Generate colormap over 256 values
    all_values = np.linspace(0, 255, 256)

    colormap = {}
    if len(colors) != len(value_points):
        raise ValueError("values and colors must be the same length")
    if isinstance(colors[0],str) and is_hex_color(colors[0]):
        colors = [hex_to_rgb(color) for color in colors]

    colors = np.array(colors)
    # Interpolate color channels
    r = np.interp(all_values, rescaled_points, colors[:, 0])
    g = np.interp(all_values, rescaled_points, colors[:, 1])
    b = np.interp(all_values, rescaled_points, colors[:, 2])
    if colors.shape[1] == 4:
        a = np.interp(all_values, rescaled_points, colors[:, 3])
    else:
        a = np.full_like(all_values, 255)
    final_colormap = {int(v): (int(r[i]), int(g[i]), int(b[i]), int(a[i])) for i, v in enumerate(all_values)}
    return final_colormap


@lru_cache(maxsize=128)
def get_tile(file_path: str, z: int, x: int, y: int, style: str = None, values: str = None, colors: str = None) -> Response:
    """
    Args:
        file_path: A string representing the path to the file.
        z: An integer representing the zoom level.
        x: An integer representing the x-coordinate of the tile.
        y: An integer representing the y-coordinate of the tile.
        style: An optional string representing the style of the tile.
        values: An optional string representing the values for the custom style.
        colors: An optional string representing the colors for the custom style.

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
                if style == 'shader_rgba':
                    img.array = data_to_rgba(img.data[0], nodata)
                    add_mask = False
                else:
                    if img.dataset_statistics:
                        img.rescale(in_range=img.dataset_statistics)
                        min_val = img.dataset_statistics[0][0]
                        max_val = img.dataset_statistics[0][1]
                    else:
                        img.rescale(in_range=[(stats.min, stats.max)])
                        min_val = stats.min
                        max_val = stats.max

                    if style == 'custom':
                        if not values or not colors:
                            raise HTTPException(status_code=400, detail='values and colors must be provided for a custom style')
                        value_points = [float(x) for x in values.split(',')]
                        if '_' in colors:
                            colors_points = [(int(x), int(y), int(z), int(a)) for x, y, z, a in
                                         [color.split(',') for color in colors.split('_')]]
                        else:
                            colors_points = [str(x) for x in colors.split(',')]
                        try:
                            colormap = generate_colormap(min_val, max_val, value_points, colors_points)
                        except ValueError as e:
                            raise HTTPException(status_code=400,
                                                detail='values and colors must be the same length. colors must be sets of 4 values r,g,b,a separated by commas and sets of colors separated by _')
                    else:
                        try:
                            colormap = get_cached_colormap(style)
                        except InvalidColorMapName:
                            raise HTTPException(status_code=404, detail=f"Invalid colormap name: {style}")
            elif img.dataset_statistics:
                img.rescale(in_range=img.dataset_statistics)
            else:
                img.rescale(in_range=[(stats.min, stats.max)])
            content = img.render(img_format="PNG", colormap=colormap, add_mask=add_mask, **img_profiles.get("png"))
            return Response(content, media_type="image/png")
    except TileOutsideBounds:
        raise HTTPException(status_code=204, detail=f"Tile out of bounds {z} {x} {y}")
