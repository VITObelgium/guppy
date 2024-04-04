from fastapi import Depends, APIRouter
from sqlalchemy.orm import Session

from guppy2.config import config as cfg
from guppy2.db import schemas as s
from guppy2.db.dependencies import get_db
from guppy2.endpoints import endpoints, endpoints_calc

router = APIRouter(
    prefix=f"{cfg.deploy.path}",
    tags=["layer management"]
)


@router.get("/layers", response_model=list[s.LayerMetadataSchema], description="Get layers mapping with pagination.")
def get_layers_mapping(db: Session = Depends(get_db), limit: int = 100, offset: int = 0):
    return endpoints.get_layers_mapping(db=db, limit=limit, offset=offset)


@router.get("/layers/{layer_name}", response_model=s.LayerMetadataSchema, description="Get mapping for a specified layer.")
def get_layer_mapping(layer_name: str, db: Session = Depends(get_db)):
    return endpoints.get_layer_mapping(db=db, layer_name=layer_name)


@router.delete("/layers/delete", description="Delete generated store for a specified layer.")
def delete_generated_store(layer: str):
    return endpoints_calc.delete_generated_store(layer)


@router.get("/healthcheck", description="Check the health status of the service.")
def healthcheck(db: Session = Depends(get_db)):
    return endpoints.healthcheck(db=db)
