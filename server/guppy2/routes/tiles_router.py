from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from starlette.responses import Response

from guppy2 import endpoints_tiles as endpoints_tiles, endpoints_rio_tiler as endpoints_rio_tiler
from guppy2.config import config as cfg
from guppy2.db.dependencies import get_db

router = APIRouter(
    prefix=f"{cfg.deploy.path}/tiles",
    tags=["tiles"]
)


@router.get("/vector/{layer_name}/{z}/{x}/{y}", description="Generate a vector tile for a specified layer.")
async def get_vector_tile(layer_name: str, z: int, x: int, y: int, db: Session = Depends(get_db)):
    return endpoints_tiles.get_tile(layer_name=layer_name, db=db, z=z, x=x, y=y)


@router.get(r"/raster/{layer_name}/{z}/{x}/{y}.png", responses={200: {"content": {"image/png": {}}, "description": "Return an image.", }}, response_class=Response,
            description="Read COG and return a png tile")
async def get_raster_tile(layer_name: str, z: int, x: int, y: int, style: str = None, db: Session = Depends(get_db)):
    return endpoints_rio_tiler.get_tile_for_layer(layer_name=layer_name, db=db, z=z, x=x, y=y, style=style)
