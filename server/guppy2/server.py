# coding: utf-8
import uvicorn
from fastapi import FastAPI, Depends, APIRouter
from sqlalchemy.orm import Session
from fastapi.responses import ORJSONResponse
import guppy2.db.schemas as s
import guppy2.endpoints as endpoints
from guppy2.config import config as cfg
from guppy2.db.db_session import SessionLocal, engine
from guppy2.db.models import Base

app = FastAPI()
api = APIRouter(prefix="/api")

Base.metadata.create_all(bind=engine)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@api.get("/layers/{layer_name}/bbox_stats", response_model=s.StatsResponse, tags=["stats"])
def get_stats_for_bbox(layer_name: str, bbox_left: float, bbox_bottom: float, bbox_right: float, bbox_top: float, native: bool = False, db: Session = Depends(get_db)):
    return endpoints.get_stats_for_bbox(db=db, layer_name=layer_name, bbox_left=bbox_left, bbox_bottom=bbox_bottom, bbox_right=bbox_right, bbox_top=bbox_top, native=native)


@api.post("/layers/{layer_name}/data", response_model=s.DataResponse, tags=["data"])
def get_data_for_wkt(layer_name: str, body: s.GeometryBody, db: Session = Depends(get_db)):
    return endpoints.get_data_for_wkt(db=db, layer_name=layer_name, body=body)


@api.post("/layers/{layer_name}/classification", response_model=s.ClassificationResult, tags=["data"])
def get_classification_for_wkt(layer_name: str, body: s.GeometryBody, db: Session = Depends(get_db)):
    return endpoints.get_classification_for_wkt(db=db, layer_name=layer_name, body=body)


@api.post("/layers/{layer_name}/line_data", response_model=s.LineDataResponse, tags=["data"])
def get_line_data_for_wkt(layer_name: str, body: s.LineGeometryBody, db: Session = Depends(get_db)):
    return endpoints.get_line_data_for_wkt(db=db, layer_name=layer_name, body=body)


@api.post("/layers/line_data", response_model=list[s.LineData], tags=["data"])
def get_line_data_list_for_wkt(body: s.LineGeometryListBody, db: Session = Depends(get_db)):
    return endpoints.get_line_data_list_for_wkt(db=db, body=body)


@api.post("/layers/multi_line_data", response_model=list[s.MultiLineData], tags=["data"])
def get_line_data_list_for_wkt(body: s.MultiLineGeometryListBody, db: Session = Depends(get_db)):
    return endpoints.get_multi_line_data_list_for_wkt(db=db, body=body)


@api.post("/layers/{layer_name}/stats", response_model=s.StatsResponse, tags=["data"])
def get_stats_for_wkt(layer_name: str, body: s.GeometryBody, native: bool = False, db: Session = Depends(get_db)):
    return endpoints.get_stats_for_wkt(db=db, layer_name=layer_name, body=body, native=native)


@api.post("/layers/statslist", response_model=list[s.StatsResponse], tags=["data"])
def get_stats_for_wkt_list(body: s.GeometryBodyList, native: bool = False, db: Session = Depends(get_db)):
    return endpoints.get_stats_for_wkt_list(db=db, body=body, native=native)


@api.get("/layers/{layer_name}/point", response_model=s.PointResponse, tags=["data"])
def get_point_value_from_raster(layer_name: str, x: float, y: float, db: Session = Depends(get_db)):
    return endpoints.get_point_value_from_raster(db=db, layer_name=layer_name, x=x, y=y)


@api.post("/layers/{layer_name}/object", tags=["data"], response_class=ORJSONResponse)
def get_line_object_list_for_wkt(layer_name: str, body: s.LineObjectGeometryBody, db: Session = Depends(get_db)):
    return endpoints.get_line_object_list_for_wkt(db=db, layer_name=layer_name, body=body)


@api.get("/layers", response_model=list[s.LayerMetadataSchema], tags=["mapping"])
def get_layers_mapping(db: Session = Depends(get_db), limit: int = 100, offset: int = 0):
    return endpoints.get_layers_mapping(db=db, limit=limit, offset=offset)


@api.get("/layers/{layer_name}", response_model=s.LayerMetadataSchema, tags=["mapping"])
def get_layer_mapping(layer_name: str, db: Session = Depends(get_db)):
    return endpoints.get_layer_mapping(db=db, layer_name=layer_name)


@api.post("/layers/contours", response_model=list[s.CountourBodyResponse], tags=["contour"])
def get_countour_for_models(body: s.CountourBodyList, db: Session = Depends(get_db)):
    return endpoints.get_countour_for_models(db=db, body=body)


@api.get("/healthcheck")
def healthcheck(db: Session = Depends(get_db)):
    return endpoints.healthcheck(db=db)


app.include_router(api)
if __name__ == '__main__':
    uvicorn.run("server:app", port=5000, reload=True)
