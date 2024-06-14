from fastapi import Depends, APIRouter
from sqlalchemy.orm import Session

from guppy.config import config as cfg
from guppy.db import schemas as s
from guppy.db.dependencies import get_db
from guppy.endpoints import endpoints

router = APIRouter(
    prefix=f"{cfg.deploy.path}/layers",
    tags=["stats"]
)


@router.get("/{layer_name}/bbox_stats", response_model=s.StatsResponse, description="Get statistics for a specified bounding box within a layer.")
def get_stats_for_bbox(layer_name: str, bbox_left: float, bbox_bottom: float, bbox_right: float, bbox_top: float, native: bool = False, db: Session = Depends(get_db)):
    return endpoints.get_stats_for_bbox(db=db, layer_name=layer_name, bbox_left=bbox_left, bbox_bottom=bbox_bottom, bbox_right=bbox_right, bbox_top=bbox_top, native=native)


@router.post("/{layer_name}/stats", response_model=s.StatsResponse, description="Get statistics for a specified wkt geometry within a layer.")
def get_stats_for_wkt(layer_name: str, body: s.GeometryBody, native: bool = False, db: Session = Depends(get_db)):
    return endpoints.get_stats_for_wkt(db=db, layer_name=layer_name, body=body, native=native)


@router.post("/{layer_name}/quantiles", response_model=s.QuantileResponse, description="Get custom quantiles for a specified wkt geometry within a layer.")
def get_quantiles_for_wkt(layer_name: str, body: s.QuantileBody, native: bool = False, db: Session = Depends(get_db)):
    return endpoints.get_quantiles_for_wkt(db=db, layer_name=layer_name, body=body, native=native)


@router.post("/statslist", response_model=list[s.StatsResponse], description="Get statistics for a given wkt geometry list.")
def get_stats_for_wkt_list(body: s.GeometryBodyList, native: bool = False, db: Session = Depends(get_db)):
    return endpoints.get_stats_for_wkt_list(db=db, body=body, native=native)
