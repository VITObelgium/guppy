from fastapi import Depends, APIRouter
from sqlalchemy.orm import Session

from guppy.config import config as cfg
from guppy.db import schemas as s
from guppy.db.dependencies import get_db
from guppy.endpoints import endpoints, endpoints_calc

router = APIRouter(
    prefix=f"{cfg.deploy.path}/layers",
    tags=["calculations"]
)


@router.post("/contours", response_model=list[s.CountourBodyResponse], description="Get countour result for specified models.")
def get_countour_for_models(body: s.CountourBodyList, db: Session = Depends(get_db)):
    return endpoints.get_countour_for_models(db=db, body=body)


@router.post("/calculate", description="Perform calculation on raster data.")
def raster_calculation(body: s.RasterCalculationBody, db: Session = Depends(get_db)):
    return endpoints_calc.raster_calculation(db=db, body=body)
