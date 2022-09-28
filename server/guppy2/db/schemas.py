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
    is_rgb: Optional[bool] = False
    rgb_factor: Optional[float] = None

    class Config:
        orm_mode = True


class PointResponse(CamelModel):
    type: str
    layer_name: str
    value: Optional[float] = None


class StatsResponse(CamelModel):
    type: str
    min: Optional[float]
    max: Optional[float]
    sum: Optional[float]
    mean: Optional[float]
    count_total: Optional[int]
    count_no_data: Optional[int]
    count_data: int
    q02: Optional[float]
    q05: Optional[float]
    q95: Optional[float]
    q98: Optional[float]


class ClassificationEntry(CamelModel):
    value: float
    count: int
    percentage: float


class ClassificationResult(CamelModel):
    type: str
    data: list[ClassificationEntry]


class DataResponse(CamelModel):
    type: str
    data: list[list[float]]


class LineDataResponse(CamelModel):
    type: str
    data: list[float]


class LineData(CamelModel):
    layer_name: str
    data: list[float]


class GeometryBody(CamelModel):
    geometry: str


class LineGeometryBody(CamelModel):
    geometry: str
    number_of_points: int


class LineGeometryListBody(CamelModel):
    geometry: str
    number_of_points: int
    layer_names: list[str]


class LineObjectGeometryBody(CamelModel):
    geometry: str
    number_of_points: int
    distance: int
