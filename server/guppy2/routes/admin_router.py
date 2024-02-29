from fastapi import Form, UploadFile, File, Depends, APIRouter
from sqlalchemy.orm import Session
from starlette.responses import FileResponse

from guppy2 import endpoints_admin as endpoints_admin
from guppy2 import endpoints_tiles as endpoints_tiles
from guppy2 import endpoints_upload as endpoints_upload
from guppy2.config import config as cfg
from guppy2.db.dependencies import get_db

router = APIRouter(
    prefix=f"{cfg.deploy.path}/admin",
    tags=["data upload"]
)


@router.post("/upload", description="Upload a file (GeoTiff or Gpkg) to the server.")
async def upload_file(layerName: str = Form(...), isRgb: bool = Form(False), file: UploadFile = File(...), db: Session = Depends(get_db)):
    return endpoints_upload.upload_file(layer_name=layerName, file=file, is_rgb=isRgb, db=db)


@router.get("/upload/ui", description="simple UI to upload a  file (GeoTiff or Gpkg) to the server.")
def read_index():
    return FileResponse('guppy2/html/index.html')


@router.delete("/layer/{layerName}", description="Delete a layer from the server.")
def delete_layer(layerName: str, db: Session = Depends(get_db)):
    return endpoints_admin.delete_layer_mapping(db=db, layer_name=layerName)


@router.put("/layer/{layerName}", description="Update a layer on the server.")
def update_layer(layerName: str, file_path: str, is_rgb: bool, is_mbtile: bool, db: Session = Depends(get_db)):
    return endpoints_admin.update_layer_mapping(db=db, layer_name=layerName, file_path=file_path, is_rgb=is_rgb, is_mbtile=is_mbtile)


@router.post("/layer", description="Insert a layer on the server.")
def insert_layer(layerName: str, file_path: str, is_rgb: bool, is_mbtile: bool, db: Session = Depends(get_db)):
    return endpoints_admin.insert_layer_mapping(db=db, layer_name=layerName, file_path=file_path, is_rgb=is_rgb, is_mbtile=is_mbtile)


@router.get("/cache/clear", description="Clear the tile cache.")
def clear_cache():
    return endpoints_tiles.clear_tile_cache()


@router.get("/tilestats", description="Get tile statistics.")
def get_tile_statistics(layerName: str, db: Session = Depends(get_db)):
    return endpoints_tiles.get_tile_statistics(db=db, layer_name=layerName)