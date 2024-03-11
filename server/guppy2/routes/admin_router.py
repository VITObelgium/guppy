from fastapi import Form, UploadFile, File, Depends, APIRouter
from fastapi.responses import HTMLResponse, Response
from sqlalchemy.orm import Session

from guppy2 import endpoints_admin as endpoints_admin
from guppy2 import endpoints_tiles as endpoints_tiles
from guppy2 import endpoints_upload as endpoints_upload
from guppy2.config import config as cfg
from guppy2.db.dependencies import get_db
from guppy2.db.schemas import LayerMetadataBody, TileStatisticsSchema

router = APIRouter(
    prefix=f"{cfg.deploy.path}/admin",
    tags=["data upload"]
)


@router.post("/upload", description="Upload a file (GeoTiff or Gpkg) to the server.")
async def upload_file(layerName: str = Form(...), isRgb: bool = Form(False), file: UploadFile = File(...), db: Session = Depends(get_db)):
    return endpoints_upload.upload_file(layer_name=layerName, file=file, is_rgb=isRgb, db=db)


@router.get("/upload/ui", description="simple UI to upload a  file (GeoTiff or Gpkg) to the server.")
def read_index():
    with open('guppy2/html/upload.html', 'r', encoding='utf-8') as file:
        file_content = file.read()

    file_content = file_content.replace('$deploy_path$', cfg.deploy.path)
    return HTMLResponse(content=file_content)


@router.get("/layers", description="layers")
async def read_index():
    with open('guppy2/html/layers.html', 'r', encoding='utf-8') as file:
        file_content = file.read()

    file_content = file_content.replace('$deploy_path$', cfg.deploy.path)
    return HTMLResponse(content=file_content)


@router.get("/stats", description="stats")
def read_index():
    with open('guppy2/html/statistics.html', 'r', encoding='utf-8') as file:
        file_content = file.read()

    file_content = file_content.replace('$deploy_path$', cfg.deploy.path)
    return HTMLResponse(content=file_content)


@router.delete("/layer/{layerName}", description="Delete a layer from the server.")
def delete_layer(layerName: str, db: Session = Depends(get_db)):
    return endpoints_admin.delete_layer_mapping(db=db, layer_name=layerName)


@router.put("/layer/{layerName}", description="Update a layer on the server.")
def update_layer(body: LayerMetadataBody, db: Session = Depends(get_db)):
    return endpoints_admin.update_layer_mapping(db=db, layer_name=body.layer_name, file_path=body.file_path, is_rgb=body.is_rgb, is_mbtile=body.is_mbtile)


@router.post("/layer", description="Insert a layer on the server.")
def insert_layer(body: LayerMetadataBody, db: Session = Depends(get_db)):
    return endpoints_admin.insert_layer_mapping(db=db, layer_name=body.layer_name, file_path=body.file_path, is_rgb=body.is_rgb, is_mbtile=body.is_mbtile)


@router.get("/cache/clear", description="Clear the tile cache.")
def clear_cache():
    return endpoints_tiles.clear_tile_cache()


@router.get("/tilestats", response_model=list[TileStatisticsSchema], description="Get tile statistics.")
def get_tile_statistics(layerName: str, offset: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    return endpoints_tiles.get_tile_statistics(db=db, layer_name=layerName, offset=offset, limit=limit)


@router.get('/tilestatsgpkg')
def get_tilestatsgpkg(layerName: str, db: Session = Depends(get_db)):
    gpkg_bytes = endpoints_tiles.get_tile_statistics_images(db=db, layer_name=layerName)
    return Response(gpkg_bytes, media_type="application/geopackage+sqlite3", headers={"Content-Disposition": f"attachment;filename={layerName}_stats.gpkg"})
