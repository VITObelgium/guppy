# coding: utf-8
import ast
from enum import Enum as PyEnum
from typing import Optional, Union

from pydantic import BaseModel, computed_field
from pydantic.fields import Field


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
    label: Optional[str] = None
    file_path: str
    data_path: Optional[str] = None
    is_rgb: Optional[bool] = False
    is_mbtile: Optional[bool] = False
    metadata_str: Optional[str] = Field(exclude=True, default=None)

    @computed_field
    @property
    def metadata(self) -> dict:
        try:
            return ast.literal_eval(self.metadata_str) if self.metadata_str else {}
        except ValueError:
            return {"value": self.metadata_str}

    @computed_field
    @property
    def effective_label(self) -> str:
        return self.label if self.label is not None and self.label else self.layer_name

    class Config:
        from_attributes = True


class PointResponse(CamelModel):
    type: str
    layer_name: str
    value: Optional[float | dict] = None


class StatsResponse(CamelModel):
    type: str
    layer_name: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    sum: Optional[float] = None
    mean: Optional[float] = None
    count_total: Optional[int] = None
    count_no_data: Optional[int] = None
    count_data: int
    q02: Optional[float] = None
    q05: Optional[float] = None
    q95: Optional[float] = None
    q98: Optional[float] = None


class QuantileList(CamelModel):
    quantile: float
    value: float


class QuantileResponse(CamelModel):
    type: str
    layer_name: Optional[str] = None
    quantiles: list[QuantileList]


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


class QuantileBody(CamelModel):
    geometry: str
    srs: Optional[str] = None
    quantiles: list[float]


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
    round_val: Optional[int] = None


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
    normalize = "normalize"
    max_normalize = "max_normalize"


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
    file_response: Optional[bool] = True
    rgb: Optional[bool] = False
    rescale_result: Optional[RescaleResult] = None
    layer_list_after_rescale: Optional[list[CombineLayersList]] = None
    result_style: Optional[str] = None
    result_label: Optional[str] = None


class LayerMetadataBody(CamelModel):
    layer_name: str
    label: str
    file_path: str
    data_path: Optional[str] = None
    is_rgb: Optional[bool] = False
    is_mbtile: Optional[bool] = False
    metadata_str: Optional[str] = Field(exclude=True, default=None)

    @computed_field
    @property
    def metadata(self) -> dict:
        return ast.literal_eval(self.metadata_str) if self.metadata_str else {}


class LayerMetadataPostBody(CamelModel):
    layer_name: str
    label: str
    file_path: str
    data_path: Optional[str] = None
    is_rgb: Optional[bool] = False
    is_mbtile: Optional[bool] = False
    metadata: Optional[str] = None


class TileStatisticsSchema(CamelModel):
    id: int
    layer_name: str
    x: int
    y: int
    z: int
    count: int

    class Config:
        from_attributes = True


class QueryParams(CamelModel):
    cql_filter: str
