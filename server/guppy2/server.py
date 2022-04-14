# coding: utf-8

import uvicorn
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

import guppy2.db.schemas as s
import guppy2.endpoints as endpoints
from guppy2.auth import get_current_user
from guppy2.config import config as cfg
from guppy2.db.db_session import SessionLocal, engine
from guppy2.db.models import Base

app = FastAPI()
api = FastAPI()

app.mount(f"{cfg.deploy.path}", api)
Base.metadata.create_all(bind=engine)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@api.get("layers/{layer_name}/bbox_stats", response_model=s.StatsResponse, tags=["stats"])
async def get_stats_for_bbox(layer_name: str, bbox_left: float, bbox_bottom: float, bbox_right: float, bbox_top: float,
                             db: Session = Depends(get_db), token_data: [str] = Depends(get_current_user)):
    return endpoints.get_stats_for_bbox(db=db, layer_name=layer_name,
                                        bbox_left=bbox_left, bbox_bottom=bbox_bottom, bbox_right=bbox_right, bbox_top=bbox_top,
                                        token_data=token_data)


@api.get("layers/{layer_name}/point", response_model=s.PointResponse, tags=["data"])
async def get_point_value_from_raster(layer_name: str, x: float, y: float, db: Session = Depends(get_db), token_data: [str] = Depends(get_current_user)):
    return endpoints.get_point_value_from_raster(db=db, layer_name=layer_name, x=x, y=y, token_data=token_data)


@api.get("/healthcheck")
async def healthcheck(db: Session = Depends(get_db)):
    return endpoints.healthcheck(db=db)


if __name__ == '__main__':
    uvicorn.run("server:app", reload=True)
