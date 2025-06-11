# coding: utf-8
import logging
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator

from guppy.config import config as cfg
from guppy.db.db_session import Base, engine
from guppy.db.db_sync import keep_db_tables_in_sync
from guppy.endpoints.tile_utils import save_request_counts, save_request_counts_timer
from guppy.routes.admin_router import router as admin_router
from guppy.routes.calculation_router import router as calculation_router
from guppy.routes.data_router import router as data_router
from guppy.routes.general_router import router as general_router
from guppy.routes.stats_router import router as stats_router
from guppy.routes.tiles_router import router as tiles_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)

keep_db_tables_in_sync()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Executes a context manager for the lifespan of the FastAPI application. This method is used to capture and save request counts in the database when the application is shutdown.
    Args:
        app: Instance of FastAPI application.

    """
    start_background_task()
    yield
    # save counts in db on shutdown
    save_request_counts()


app = FastAPI(
    title="guppy",
    description="A raster analyzer API",
    docs_url=f"{cfg.deploy.path}/docs",
    openapi_url=f"{cfg.deploy.path}",
    lifespan=lifespan,
)
ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE"]

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=ALLOWED_METHODS,  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def start_background_task():
    """
    Starts a background task that runs the save_request_counts_timer function in a separate thread.

    This method creates a new thread and starts it with the save_request_counts_timer function as the target.
    The thread is set as a daemon thread, which means it will automatically stop when the main thread exits.

    :return: None
    """
    thread = threading.Thread(target=save_request_counts_timer, daemon=True)
    thread.start()


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("guppy/html/favicon.ico")


app.include_router(general_router)
app.include_router(tiles_router)
app.include_router(data_router)
app.include_router(stats_router)
app.include_router(calculation_router)
app.include_router(admin_router)

instrumentator = Instrumentator()
instrumentator.instrument(app)
instrumentator.expose(app, endpoint=f"{cfg.deploy.path}/admin/metrics")

if __name__ == "__main__":
    uvicorn.run(app, port=5000)
