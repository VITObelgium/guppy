import hashlib
import logging
import os
import sqlite3
from contextlib import closing

import numpy as np
import rasterio
from fastapi import HTTPException
from rasterio.enums import Resampling
from rio_tiler.utils import render, resize_array
from sqlalchemy.orm import Session
from starlette.responses import Response

from guppy.db.models import LayerMetadata

logger = logging.getLogger(__name__)
MBTILES_OVERVIEW_ZOOM = 12


def _get_layer_source(db: Session, layer_name: str) -> tuple[str, bool]:
    layer = db.query(LayerMetadata).filter_by(layer_name=layer_name).first()
    if not layer:
        raise HTTPException(status_code=404, detail=f"Layer not found: {layer_name}")

    file_path = layer.file_path
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    return file_path, bool(layer.is_mbtile)


def _fitted_size(source_width: int, source_height: int, width: int, height: int) -> tuple[int, int]:
    scale = min(width / source_width, height / source_height)
    return (
        max(1, min(width, round(source_width * scale))),
        max(1, min(height, round(source_height * scale))),
    )


def _render_cog_overview(file_path: str, width: int, height: int) -> bytes:
    try:
        with rasterio.open(file_path) as source:
            render_width, render_height = _fitted_size(source.width, source.height, width, height)
            data = source.read(
                1,
                out_shape=(render_height, render_width),
                masked=True,
                resampling=Resampling.average,
            )
    except rasterio.errors.RasterioError as exc:
        raise HTTPException(status_code=422, detail=f"Could not read COG: {exc}") from exc

    values = np.asarray(data.data, dtype=np.float64)
    valid = ~np.ma.getmaskarray(data) & np.isfinite(values)
    grayscale = np.full((render_height, render_width), 255, dtype=np.uint8)

    if np.any(valid):
        minimum = float(np.min(values[valid]))
        maximum = float(np.max(values[valid]))
        if maximum > minimum:
            scaled = np.clip((values[valid] - minimum) * 255.0 / (maximum - minimum), 0, 255)
            grayscale[valid] = scaled.astype(np.uint8)
        else:
            grayscale[valid] = 0

        rows, columns = np.nonzero(valid)
        grayscale = grayscale[rows[0]:rows[-1] + 1, columns.min():columns.max() + 1]
        render_width, render_height = _fitted_size(
            grayscale.shape[1], grayscale.shape[0], width, height
        )
        if grayscale.shape != (render_height, render_width):
            grayscale = resize_array(
                grayscale, render_height, render_width, resampling_method="bilinear"
            )

    canvas = np.full((3, height, width), 255, dtype=np.uint8)
    x_offset = (width - render_width) // 2
    y_offset = (height - render_height) // 2
    canvas[:, y_offset:y_offset + render_height, x_offset:x_offset + render_width] = grayscale
    return render(canvas, img_format="PNG")


def _draw_tile_outline(canvas: np.ndarray, left: int, top: int, right: int, bottom: int) -> None:
    canvas[:, top, left:right + 1] = 0
    canvas[:, bottom, left:right + 1] = 0
    canvas[:, top:bottom + 1, left] = 0
    canvas[:, top:bottom + 1, right] = 0


def _render_mbtiles_overview(file_path: str, width: int, height: int) -> bytes:
    try:
        uri = f"file:{file_path}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            rows = connection.execute(
                "SELECT tile_column, tile_row FROM tiles WHERE zoom_level = ?",
                (MBTILES_OVERVIEW_ZOOM,),
            ).fetchall()
    except sqlite3.Error as exc:
        raise HTTPException(status_code=422, detail=f"Could not read MBTiles: {exc}") from exc

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No tiles found at zoom level {MBTILES_OVERVIEW_ZOOM}",
        )

    # MBTiles rows use TMS (bottom-up); convert them to XYZ/display rows (top-down).
    tiles = [(column, (1 << MBTILES_OVERVIEW_ZOOM) - 1 - row) for column, row in rows]
    min_column = min(column for column, _ in tiles)
    max_column = max(column for column, _ in tiles)
    min_row = min(row for _, row in tiles)
    max_row = max(row for _, row in tiles)
    column_count = max_column - min_column + 1
    row_count = max_row - min_row + 1
    render_width, render_height = _fitted_size(column_count, row_count, width, height)
    x_offset = (width - render_width) // 2
    y_offset = (height - render_height) // 2

    canvas = np.full((3, height, width), 255, dtype=np.uint8)
    for column, row in tiles:
        left = x_offset + round((column - min_column) * render_width / column_count)
        right = x_offset + round((column - min_column + 1) * render_width / column_count) - 1
        top = y_offset + round((row - min_row) * render_height / row_count)
        bottom = y_offset + round((row - min_row + 1) * render_height / row_count) - 1
        left = min(max(left, 0), width - 1)
        right = min(max(right, left), width - 1)
        top = min(max(top, 0), height - 1)
        bottom = min(max(bottom, top), height - 1)
        _draw_tile_outline(canvas, left, top, right, bottom)

    return render(canvas, img_format="PNG")


def get_layer_overview(layer_name: str, width: int, height: int, db: Session) -> Response:
    file_path, is_mbtile = _get_layer_source(db, layer_name)
    logger.info(f"Rendering overview for layer {layer_name}")
    if is_mbtile:
        content = _render_mbtiles_overview(file_path, width, height)
    else:
        content = _render_cog_overview(file_path, width, height)

    return Response(
        content,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=300, must-revalidate",
            "ETag": f'"{hashlib.sha256(content).hexdigest()}"',
        },
    )
