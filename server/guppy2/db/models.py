# coding: utf-8
from sqlalchemy import Column, Integer, String, text, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class LayerMetadata(Base):
    __tablename__ = 'layer_metadata'
    __table_args__ = {'schema': 'guppy2'}

    id = Column(Integer, primary_key=True, server_default=text("nextval('guppy2.layer_metadata_seq'::regclass)"))
    layer_name = Column(String, nullable=False, unique=True)
    file_path = Column(String, nullable=False)
    is_rgb = Column(Boolean)
    rgb_factor = Column(Float)

