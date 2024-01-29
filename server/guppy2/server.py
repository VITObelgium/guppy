# coding: utf-8
import logging

from fastapi import FastAPI, Depends, APIRouter, UploadFile, File, Form
from fastapi.responses import ORJSONResponse, FileResponse, Response
from sqlalchemy.orm import Session

import guppy2.db.schemas as s
import guppy2.endpoints as endpoints
import guppy2.endpoints_calc as endpoints_calc
import guppy2.endpoints_tiles as endpoints_tiles
import guppy2.endpoints_upload as endpoints_upload
from guppy2.config import config as cfg
from guppy2.db.db_session import SessionLocal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(docs_url=f"{cfg.deploy.path}/docs", openapi_url=f"{cfg.deploy.path}")
api = APIRouter(prefix=f"{cfg.deploy.path}")


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@api.get("/layers/{layer_name}/bbox_stats", response_model=s.StatsResponse, tags=["stats"], description="Get statistics for a specified bounding box within a layer.")
def get_stats_for_bbox(layer_name: str, bbox_left: float, bbox_bottom: float, bbox_right: float, bbox_top: float, native: bool = False, db: Session = Depends(get_db)):
    return endpoints.get_stats_for_bbox(db=db, layer_name=layer_name, bbox_left=bbox_left, bbox_bottom=bbox_bottom, bbox_right=bbox_right, bbox_top=bbox_top, native=native)


@api.post("/layers/{layer_name}/data", response_model=s.DataResponse, tags=["data"], description="Get data for a specified wkt geometry within a layer.")
def get_data_for_wkt(layer_name: str, body: s.GeometryBody, db: Session = Depends(get_db)):
    return endpoints.get_data_for_wkt(db=db, layer_name=layer_name, body=body)


@api.post("/layers/{layer_name}/classification", response_model=s.ClassificationResult, tags=["data"], description="Get classification result for a specified wkt geometry within a layer.")
def get_classification_for_wkt(layer_name: str, body: s.GeometryBody, db: Session = Depends(get_db)):
    return endpoints.get_classification_for_wkt(db=db, layer_name=layer_name, body=body)


@api.post("/layers/{layer_name}/line_data", response_model=s.LineDataResponse, tags=["data"], description="Get line data for a specified wkt geometry within a layer.")
def get_line_data_for_wkt(layer_name: str, body: s.LineGeometryBody, db: Session = Depends(get_db)):
    return endpoints.get_line_data_for_wkt(db=db, layer_name=layer_name, body=body)


@api.post("/layers/line_data", response_model=list[s.LineData], tags=["data"], description="Get line data list for a given wkt geometry list.")
def get_line_data_list_for_wkt(body: s.LineGeometryListBody, db: Session = Depends(get_db)):
    return endpoints.get_line_data_list_for_wkt(db=db, body=body)


@api.post("/layers/multi_line_data", response_model=list[s.MultiLineData], tags=["data"], description="Get multi-line data list for a given wkt geometry list.")
def get_line_data_list_for_wkt(body: s.MultiLineGeometryListBody, db: Session = Depends(get_db)):
    return endpoints.get_multi_line_data_list_for_wkt(db=db, body=body)


@api.post("/layers/{layer_name}/stats", response_model=s.StatsResponse, tags=["stats"], description="Get statistics for a specified wkt geometry within a layer.")
def get_stats_for_wkt(layer_name: str, body: s.GeometryBody, native: bool = False, db: Session = Depends(get_db)):
    return endpoints.get_stats_for_wkt(db=db, layer_name=layer_name, body=body, native=native)


@api.post("/layers/statslist", response_model=list[s.StatsResponse], tags=["stats"], description="Get statistics for a given wkt geometry list.")
def get_stats_for_wkt_list(body: s.GeometryBodyList, native: bool = False, db: Session = Depends(get_db)):
    return endpoints.get_stats_for_wkt_list(db=db, body=body, native=native)


@api.get("/layers/{layer_name}/point", response_model=s.PointResponse, tags=["data"], description="Get point value for a given coordinate from raster within a layer.")
def get_point_value_from_raster(layer_name: str, x: float, y: float, db: Session = Depends(get_db)):
    return endpoints.get_point_value_from_raster(db=db, layer_name=layer_name, x=x, y=y)


@api.post("/layers/{layer_name}/object", tags=["data"], response_class=ORJSONResponse, description="Get object list for a given line wkt geometry within a layer.")
def get_line_object_list_for_wkt(layer_name: str, body: s.LineObjectGeometryBody, db: Session = Depends(get_db)):
    return endpoints.get_line_object_list_for_wkt(db=db, layer_name=layer_name, body=body)


@api.get("/layers", response_model=list[s.LayerMetadataSchema], tags=["mapping"], description="Get layers mapping with pagination.")
def get_layers_mapping(db: Session = Depends(get_db), limit: int = 100, offset: int = 0):
    return endpoints.get_layers_mapping(db=db, limit=limit, offset=offset)


@api.get("/layers/{layer_name}", response_model=s.LayerMetadataSchema, tags=["mapping"], description="Get mapping for a specified layer.")
def get_layer_mapping(layer_name: str, db: Session = Depends(get_db)):
    return endpoints.get_layer_mapping(db=db, layer_name=layer_name)


@api.post("/layers/contours", response_model=list[s.CountourBodyResponse], tags=["contour"], description="Get countour result for specified models.")
def get_countour_for_models(body: s.CountourBodyList, db: Session = Depends(get_db)):
    return endpoints.get_countour_for_models(db=db, body=body)


@api.post("/layers/calculate", tags=["calculation"], description="Perform calculation on raster data.")
def raster_calculation(body: s.RasterCalculationBody, db: Session = Depends(get_db)):
    return endpoints_calc.raster_calculation(db=db, body=body)


@api.delete("/layers/delete", tags=["calculation"], description="Delete generated store for a specified layer.")
def delete_generated_store(layer: str):
    return endpoints_calc.delete_generated_store(layer)


@api.get("/healthcheck", description="Check the health status of the service.")
def healthcheck(db: Session = Depends(get_db)):
    return endpoints.healthcheck(db=db)


@api.post("/upload", tags=["data upload"], description="Upload a raster file (GeoTiff or Ascii) to the server.")
async def upload_file(layerName: str = Form(...), isRgb: bool = Form(False), file: UploadFile = File(...), db: Session = Depends(get_db)):
    return endpoints_upload.upload_file(layer_name=layerName, file=file, is_rgb=isRgb, db=db)


@api.get("/uploadUi", tags=["data upload"], description="Upload a raster file (GeoTiff or Ascii) to the server.")
async def read_index():
    return FileResponse('guppy2/html/index.html')


@api.get("/tiles/{layer_name}/{z}/{x}/{y}")
async def get_tile(layer_name: str, z: int, x: int, y: int, db: Session = Depends(get_db)):
    tile = endpoints_tiles.get_tile(layer_name=layer_name, db=db, z=z, x=x, y=y)
    headers = {
        "Access-Control-Allow-Origin": "*",  # Replace '*' with your specific origin if needed
        "Access-Control-Allow-Methods": "GET",
        "Access-Control-Allow-Headers": "Authorization, Content-Type",
    }
    return Response(content=tile, headers=headers)
