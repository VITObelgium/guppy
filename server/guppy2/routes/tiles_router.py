from fastapi import APIRouter, Depends, Query
from rio_tiler.colormap import cmap
from sqlalchemy.orm import Session
from starlette.responses import Response

from guppy2.config import config as cfg
from guppy2.db.dependencies import get_db
from guppy2.endpoints import endpoints_rio_tiler, endpoints_tiles

router = APIRouter(
    prefix=f"{cfg.deploy.path}/tiles",
    tags=["tiles"]
)


@router.get("/vector/{layer_name}/{z}/{x}/{y}", description="Generate a vector tile for a specified layer.")
async def get_vector_tile(layer_name: str, z: int, x: int, y: int, db: Session = Depends(get_db)):
    return endpoints_tiles.get_tile(layer_name=layer_name, db=db, z=z, x=x, y=y)


@router.get(r"/raster/{layer_name}/{z}/{x}/{y}.png", responses={200: {"content": {"image/png": {}}, "description": "Return an image.", }}, response_class=Response,
            description="Read COG and return a png tile")
async def get_raster_tile(layer_name: str, z: int, x: int, y: int,
                          style: str = Query(..., description=f"Style should be 'shader_rgba' or one of {list(cmap.data.keys())} values"),
                          db: Session = Depends(get_db)):
    return endpoints_rio_tiler.get_tile_for_layer(layer_name=layer_name, db=db, z=z, x=x, y=y, style=style)


@router.get("/vector/{layer_name}/search", description="Search for a vector tile for a specified layer.")
async def search_vector_tile(layer_name: str, fieldName: str, searchString: str, zoomLevel: int = 12, db: Session = Depends(get_db)):
    return endpoints_tiles.search_tile(layer_name=layer_name, field_name=fieldName, search_string=searchString, zoom_level=zoomLevel, db=db)
