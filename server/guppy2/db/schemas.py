# coding: utf-8

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
    value: float


class StatsResponse(CamelModel):
    type: str
    min: float
    max: float
    sum: float
    mean: float
    count: float
    q02: float
    q05: float
    q95: float
    q98: float
