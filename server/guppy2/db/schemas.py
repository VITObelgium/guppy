# coding: utf-8
from typing import Optional

from pydantic import BaseModel


def to_camel(s):
    parts = iter(s.split("_"))
    return next(parts) + "".join(i.title() for i in parts)


class CamelModel(BaseModel):
    class Config:
        alias_generator = to_camel
        allow_population_by_field_name = True


class LayerMetadataSchema(CamelModel):
    id: int
    layer_name: str
    file_path: str

    class Config:
        orm_mode = True


class PointResponse(CamelModel):
    type: str
    layer_name: str
    value: Optional[float] = None


class StatsResponse(CamelModel):
    type: str
    min: float
    max: float
    sum: float
    mean: float
    count: int
    q02: float
    q05: float
    q95: float
    q98: float


class DataResponse(CamelModel):
    type: str
    data: list[list[float]]


class LineDataResponse(CamelModel):
    type: str
    data: list[float]


class GeometryBody(CamelModel):
    geometry: str


class LineGeometryBody(CamelModel):
    geometry: str
    number_of_points: int
