from fastapi import APIRouter, Depends, Query
from rio_tiler.colormap import cmap
from sqlalchemy.orm import Session
from starlette.responses import Response

from guppy.config import config as cfg
from guppy.db.dependencies import get_db
from guppy.db.schemas import QueryParams
from guppy.endpoints import endpoints_rio_tiler, endpoints_tiles

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
                          style: str = Query(..., description=f"Style should be '<b>shader_rgba</b>', '<b>custom</b>' or one of {list(cmap.data.keys())} values. <br><br>"
                                                              f"If '<b>custom</b>', extra parameters values and colors are needed like:<br> values=1.23,80.35,190.587&colors=255,0,0,255_0,255,0,255_0,0,255,255 <br>"
                                                              f"so values are comma seperated, and colors are r,g,b,a and _ seperated."),
                          values: str = None, colors: str = None,
                          db: Session = Depends(get_db)):
    return endpoints_rio_tiler.get_tile_for_layer(layer_name=layer_name, db=db, z=z, x=x, y=y, style=style, values=values, colors=colors)


@router.post("/vector/{layer_name}/search", description="Search for a vector tile for a specified layer.")
async def search_vector_tile(layer_name: str, params: QueryParams, limit: int = 100, offset: int = 0, db: Session = Depends(get_db)):
    return endpoints_tiles.search_tile(layer_name=layer_name, params=params, limit=limit, offset=offset, db=db)
