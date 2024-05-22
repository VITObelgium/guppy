from fastapi import Depends, APIRouter
from fastapi.responses import ORJSONResponse
from sqlalchemy.orm import Session

from guppy.endpoints import endpoints
from guppy.config import config as cfg
from guppy.db import schemas as s
from guppy.db.dependencies import get_db

router = APIRouter(
    prefix=f"{cfg.deploy.path}/layers",
    tags=["data"]
)


@router.post("/{layer_name}/data", response_model=s.DataResponse, description="Get data for a specified wkt geometry within a layer.")
def get_data_for_wkt(layer_name: str, body: s.GeometryBody, db: Session = Depends(get_db)):
    return endpoints.get_data_for_wkt(db=db, layer_name=layer_name, body=body)


@router.post("/{layer_name}/classification", response_model=s.ClassificationResult, description="Get classification result for a specified wkt geometry within a layer.")
def get_classification_for_wkt(layer_name: str, body: s.GeometryBody, db: Session = Depends(get_db)):
    return endpoints.get_classification_for_wkt(db=db, layer_name=layer_name, body=body)


@router.post("/{layer_name}/line_data", response_model=s.LineDataResponse, description="Get line data for a specified wkt geometry within a layer.")
def get_line_data_for_wkt(layer_name: str, body: s.LineGeometryBody, db: Session = Depends(get_db)):
    return endpoints.get_line_data_for_wkt(db=db, layer_name=layer_name, body=body)


@router.post("/line_data", response_model=list[s.LineData], description="Get line data list for a given wkt geometry list.")
def get_line_data_list_for_wkt(body: s.LineGeometryListBody, db: Session = Depends(get_db)):
    return endpoints.get_line_data_list_for_wkt(db=db, body=body)


@router.post("/multi_line_data", response_model=list[s.MultiLineData], description="Get multi-line data list for a given wkt geometry list.")
def get_multi_line_data_list_for_wkt(body: s.MultiLineGeometryListBody, db: Session = Depends(get_db)):
    return endpoints.get_multi_line_data_list_for_wkt(db=db, body=body)


@router.get("/{layer_name}/point", response_model=s.PointResponse, description="Get point value for a given coordinate from raster within a layer.")
def get_point_value_from_raster(layer_name: str, x: float, y: float, db: Session = Depends(get_db)):
    return endpoints.get_point_value_from_raster(db=db, layer_name=layer_name, x=x, y=y)


@router.post("/{layer_name}/object", response_class=ORJSONResponse, description="Get object list for a given line wkt geometry within a layer.")
def get_line_object_list_for_wkt(layer_name: str, body: s.LineObjectGeometryBody, db: Session = Depends(get_db)):
    return endpoints.get_line_object_list_for_wkt(db=db, layer_name=layer_name, body=body)
