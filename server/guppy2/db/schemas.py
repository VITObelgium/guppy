# coding: utf-8
from enum import Enum as PyEnum
from typing import Optional, Union

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
    is_mbtile: Optional[bool] = False

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
    geometry: Optional[str] = None
    number_of_points: int
    distance: int


class AllowedOperations(str, PyEnum):
    add = "add"
    subtract = "subtract"
    multiply = "multiply"
    invert_boolean_mask = "invert_boolean_mask"
    boolean_mask = "boolean_mask"
    unique_product = "unique_product"
    clip = "clip"


class CombineLayersList(CamelModel):
    layer_name: str
    operation: AllowedOperations
    operation_data: Optional[list[float]] = None
    factor: Optional[float] = 1


class CombineLayersGeometryBody(CamelModel):
    layer_list: list[CombineLayersList]
    geometry: str


class CountourBodyList(CamelModel):
    layer_names: list[str]


class CountourBodyResponse(CamelModel):
    layer_name: str
    geometry: list[dict]


class AllowedRescaleTypes(str, PyEnum):
    quantile = "quantile"
    natural_breaks = "natural breaks"
    equal_interval = "equal interval"
    provided = "provided"


class RescaleResult(CamelModel):
    rescale_type: AllowedRescaleTypes
    breaks: Union[list[float], dict]
    filter_value: Optional[float] = None
    clip_positive: Optional[bool] = False


class RasterCalculationBody(CamelModel):
    layer_list: list[CombineLayersList]
    geoserver: Optional[bool] = False
    rgb: Optional[bool] = False
    rescale_result: Optional[RescaleResult] = None
    layer_list_after_rescale: Optional[list[CombineLayersList]] = None
    result_style: Optional[str] = None


class LayerMetadataBody(CamelModel):
    layer_name: str
    file_path: str
    is_rgb: Optional[bool] = False
    is_mbtile: Optional[bool] = False

class TileStatisticsSchema(CamelModel):
    id: int
    layer_name: str
    x: int
    y: int
    z: int
    count: int

    class Config:
        from_attributes = True