from fastapi import Form, UploadFile, File, Depends, APIRouter
from sqlalchemy.orm import Session
from starlette.responses import FileResponse

from guppy2 import endpoints_upload as endpoints_upload
from guppy2.config import config as cfg
from guppy2.db.dependencies import get_db

router = APIRouter(
    prefix=f"{cfg.deploy.path}/upload",
    tags=["data upload"]
)


@router.post("", description="Upload a file (GeoTiff or Gpkg) to the server.")
async def upload_file(layerName: str = Form(...), isRgb: bool = Form(False), file: UploadFile = File(...), db: Session = Depends(get_db)):
    return endpoints_upload.upload_file(layer_name=layerName, file=file, is_rgb=isRgb, db=db)


@router.get("/ui", description="simple UI to upload a  file (GeoTiff or Gpkg) to the server.")
async def read_index():
    return FileResponse('/html/index.html')
