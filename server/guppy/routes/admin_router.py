from fastapi import Form, UploadFile, File, Depends, APIRouter
from fastapi.responses import HTMLResponse, Response
from sqlalchemy.orm import Session

from guppy.config import config as cfg
from guppy.db.dependencies import get_db
from guppy.db.schemas import LayerMetadataBody, TileStatisticsSchema, LayerMetadataPostBody
from guppy.endpoints import endpoints_admin, endpoints_upload, endpoints_tiles

router = APIRouter(
    prefix=f"{cfg.deploy.path}/admin",
    tags=["data upload"]
)


@router.post("/upload", description="Upload a file (GeoTiff or Gpkg) to the server.")
def upload_file(layerName: str = Form(...), layerLabel: str = Form(...), metadata: str = Form(None), isRgb: bool = Form(False), maxZoom: int = Form(17), file: UploadFile = File(...),
                data: UploadFile = File(None),
                db: Session = Depends(get_db)):
    return endpoints_upload.upload_file(layer_name=layerName, label=layerLabel, file=file, data=data, is_rgb=isRgb, max_zoom=maxZoom, db=db, metadata=metadata)


@router.get("/{layer_name}/generate_db", description="Generate sqlite file for a mbtiles layer")
def upload_file(layer_name: str, db: Session = Depends(get_db)):
    return endpoints_upload.generate_sqlite_file(layer_name=layer_name, db=db)


@router.get("/upload/ui", description="simple UI to upload a  file (GeoTiff or Gpkg) to the server.")
def read_index():
    with open(f'guppy/html/upload.html', 'r', encoding='utf-8') as file:
        file_content = file.read()

    file_content = file_content.replace('$deploy_path$', cfg.deploy.path)
    return HTMLResponse(content=file_content)


@router.get("/layers", description="layers")
def read_index():
    with open(f'guppy/html/layers.html', 'r', encoding='utf-8') as file:
        file_content = file.read()

    file_content = file_content.replace('$deploy_path$', cfg.deploy.path)
    return HTMLResponse(content=file_content)


@router.get("/stats", description="stats")
def read_index():
    with open(f'guppy/html/statistics.html', 'r', encoding='utf-8') as file:
        file_content = file.read()

    file_content = file_content.replace('$deploy_path$', cfg.deploy.path)
    return HTMLResponse(content=file_content)


@router.delete("/layer/{layerName}", description="Delete a layer from the server.")
def delete_layer(layerName: str, db: Session = Depends(get_db)):
    return endpoints_admin.delete_layer_mapping(db=db, layer_name=layerName)


@router.put("/layer/{layerName}", description="Update a layer on the server.")
def update_layer(body: LayerMetadataPostBody, db: Session = Depends(get_db)):
    return endpoints_admin.update_layer_mapping(db=db, layer_name=body.layer_name, label=body.label, file_path=body.file_path, data_path=body.data_path, metadata=body.metadata, is_rgb=body.is_rgb,
                                                is_mbtile=body.is_mbtile)


@router.post("/layer", description="Insert a layer on the server.")
def insert_layer(body: LayerMetadataPostBody, db: Session = Depends(get_db)):
    return endpoints_admin.insert_layer_mapping(db=db, layer_name=body.layer_name, label=body.label, file_path=body.file_path, data_path=body.data_path, metadata=body.metadata, is_rgb=body.is_rgb,
                                                is_mbtile=body.is_mbtile)


@router.get("/cache/clear", description="Clear the tile cache.")
def clear_cache():
    return endpoints_tiles.clear_tile_cache()


@router.get("/tilestats", response_model=list[TileStatisticsSchema], description="Get tile statistics.")
def get_tile_statistics(layerName: str, offset: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    return endpoints_tiles.get_tile_statistics(db=db, layer_name=layerName, offset=offset, limit=limit)


@router.get('/tilestatsgpkg', description="Returns a GPKG file with the tile cache results.")
def get_tilestatsgpkg(layerName: str, db: Session = Depends(get_db)):
    gpkg_bytes = endpoints_tiles.get_tile_statistics_images(db=db, layer_name=layerName)
    return Response(gpkg_bytes, media_type="application/geopackage+sqlite3", headers={"Content-Disposition": f"attachment;filename={layerName}_stats.gpkg"})


@router.get("/map", description="map")
def read_index():
    with open(f'guppy/html/map.html', 'r', encoding='utf-8') as file:
        file_content = file.read()

    file_content = file_content.replace('$deploy_path$', cfg.deploy.path)
    return HTMLResponse(content=file_content)
