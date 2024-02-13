from fastapi import Depends, APIRouter
from sqlalchemy.orm import Session

from guppy2 import endpoints as endpoints, endpoints_calc as endpoints_calc
from guppy2.config import config as cfg
from guppy2.db import schemas as s
from guppy2.db.dependencies import get_db

router = APIRouter(
    prefix=f"{cfg.deploy.path}/layers",
    tags=["calculations"]
)


@router.post("/contours", response_model=list[s.CountourBodyResponse], tags=["contour"], description="Get countour result for specified models.")
def get_countour_for_models(body: s.CountourBodyList, db: Session = Depends(get_db)):
    return endpoints.get_countour_for_models(db=db, body=body)


@router.post("/calculate", tags=["calculation"], description="Perform calculation on raster data.")
def raster_calculation(body: s.RasterCalculationBody, db: Session = Depends(get_db)):
    return endpoints_calc.raster_calculation(db=db, body=body)
