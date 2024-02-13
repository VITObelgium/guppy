# coding: utf-8
import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from guppy2.config import config as cfg
from guppy2.db.db_session import Base, engine
from guppy2.routes.calculation_router import router as calculation_router
from guppy2.routes.data_router import router as data_router
from guppy2.routes.general_router import router as general_router
from guppy2.routes.stats_router import router as stats_router
from guppy2.routes.tiles_router import router as tiles_router
from guppy2.routes.upload_router import router as upload_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)
app = FastAPI(title="guppy", description="A raster analyzer API", docs_url=f"{cfg.deploy.path}/docs", openapi_url=f"{cfg.deploy.path}")

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(general_router)
app.include_router(tiles_router)
app.include_router(upload_router)
app.include_router(data_router)
app.include_router(stats_router)
app.include_router(calculation_router)

if __name__ == '__main__':
    uvicorn.run("server:app", port=5000, workers=1)
