"""rio-tiler tile server."""

import os

from fastapi import HTTPException
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import Reader
from rio_tiler.profiles import img_profiles
from sqlalchemy.orm import Session
from starlette.responses import Response

from guppy2.db.models import LayerMetadata


def get_tile(layer_name: str, db: Session, z: int, x: int, y: int):
    layer = db.query(LayerMetadata).filter_by(layer_name=layer_name).first()
    if not layer:
        raise HTTPException(status_code=404, detail=f"Layer not found: {layer_name}")
    file_path = layer.file_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    """Handle tile requests."""
    try:
        img = None
        with Reader(file_path) as cog:
            if cog.tile_exists(x, y, z):
                img = cog.tile(x, y, z)
        if img:
            content = img.render(img_format="PNG", **img_profiles.get("png"))
            return Response(content, media_type="image/png")
    except TileOutsideBounds:
        raise HTTPException(status_code=404, detail=f"Tile out of bounds {z} {x} {y}")
