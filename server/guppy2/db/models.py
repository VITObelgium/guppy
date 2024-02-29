# coding: utf-8
from sqlalchemy import Column, Integer, String, text, Boolean

from guppy2.db.db_session import Base
from ..config import config

metadata = Base.metadata


class LayerMetadata(Base):
    __tablename__ = 'layer_metadata'
    __table_args__ = {'schema': config.database.db} if config.database.type == 'postgres' else {}

    id = Column(Integer, primary_key=True, server_default=text(f"nextval('{config.database.db}.layer_metadata_seq'::regclass)") if config.database.type == 'postgres' else None)
    layer_name = Column(String, nullable=False, unique=True)
    file_path = Column(String, nullable=False)
    is_rgb = Column(Boolean, nullable=False, default=False, server_default=text('FALSE'))
    is_mbtile = Column(Boolean, nullable=False, default=False, server_default=text('FALSE'))


class TileStatistics(Base):
    __tablename__ = 'tile_statistics'
    __table_args__ = {'schema': config.database.db} if config.database.type == 'postgres' else {}

    id = Column(Integer, primary_key=True, server_default=text(f"nextval('{config.database.db}.tile_statistics_seq'::regclass)") if config.database.type == 'postgres' else None)
    layer_name = Column(String, nullable=False)
    x = Column(Integer, nullable=False)
    y = Column(Integer, nullable=False)
    z = Column(Integer, nullable=False)
    count = Column(Integer, nullable=False)