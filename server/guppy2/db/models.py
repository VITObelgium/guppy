# coding: utf-8
from sqlalchemy import Column, Integer, String, text, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base

from ..config import config

Base = declarative_base()
metadata = Base.metadata


class LayerMetadata(Base):
    __tablename__ = 'layer_metadata'
    __table_args__ = {'schema': config.database.db} if config.database.type == 'postgres' else {}

    id = Column(Integer, primary_key=True, server_default=text(f"nextval('{config.database.db}.layer_metadata_seq'::regclass)") if config.database.type == 'postgres' else None)
    layer_name = Column(String, nullable=False, unique=True)
    file_path = Column(String, nullable=False)
    is_rgb = Column(Boolean)
    rgb_factor = Column(Float)
    is_mbtile = Column(Boolean)
