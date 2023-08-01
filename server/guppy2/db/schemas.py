# coding: utf-8
from typing import Optional
from decimal import Decimal
from enum import Enum as PyEnum
from pydantic import BaseModel


def to_camel(s):
    parts = iter(s.split("_"))
    return next(parts) + "".join(i.title() for i in parts)


class CamelModel(BaseModel):
    class Config:
        alias_generator = to_camel
        populate_by_name = True


class LayerMetadataSchema(CamelModel):
    id: int
    layer_name: str
    file_path: str
    is_rgb: Optional[bool] = False
    rgb_factor: Optional[float] = None

    class Config:
        from_attributes = True


class PointResponse(CamelModel):
    type: str
    layer_name: str
    value: Optional[float] = None


class StatsResponse(CamelModel):
    type: str
    layer_name: Optional[str]
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


class MultiLineData(CamelModel):
    key: str
    line_data: list[LineData]


class GeometryBody(CamelModel):
    geometry: str
    srs: Optional[str] = None


class GeometryBodyList(CamelModel):
    layer_names: list[str]
    geometry: str
    srs: Optional[str] = None


class LineGeometryBody(CamelModel):
    geometry: str
    number_of_points: int
    srs: Optional[str] = None


class LineGeometryListBody(CamelModel):
    geometry: str
    number_of_points: int
    layer_names: list[str]


class MultiLineGeometryListBody(CamelModel):
    geometry: list[str]
    number_of_points: int
    layer_names: list[str]
    round_val: Optional[int]


class LineObjectGeometryBody(CamelModel):
    geometry: Optional[str]
    number_of_points: int
    distance: int


class AllowedOperations(str, PyEnum):
    add = "add"
    subtract = "subtract"
    multiply = "multiply"


class CombineLayersList(CamelModel):
    layer_name: str
    operation: AllowedOperations
    factor: Optional[float] = 1


class CombineLayersGeometryBody(CamelModel):
    layer_list: list[CombineLayersList]
    geometry: str


class CountourBodyList(CamelModel):
    layer_names: list[str]


class CountourBodyResponse(CamelModel):
    layer_name: str
    geometry: list[dict]


class RasterCalculationBody(CamelModel):
    layer_list: list[CombineLayersList]
    geoserver: Optional[bool] = False
