import logging

from sqlalchemy import MetaData, text
from sqlalchemy.schema import CreateSchema

from guppy.config import config
from guppy.db.db_session import Base, engine
from guppy.db.models import LayerMetadata, TileStatistics

logger = logging.getLogger(__name__)


def initialize_db():
    with engine.begin() as conn:
        if config.database.type == "postgres":
            conn.execute(CreateSchema(config.database.schema, if_not_exists=True))
        Base.metadata.create_all(bind=conn)


def keep_db_tables_in_sync():
    schema = config.database.schema if config.database.type == "postgres" else None
    metadata = MetaData(schema=schema)
    metadata.reflect(bind=engine)
    for table_model in [LayerMetadata, TileStatistics]:
        table = metadata.tables[table_model.__table__.key]

        for column in table_model.__table__.columns:
            if column.name not in [c.name for c in table.columns]:
                column_type = column.type.compile(dialect=engine.dialect)
                preparer = engine.dialect.identifier_preparer
                table_name = preparer.format_table(table_model.__table__)
                column_name = preparer.quote(column.name)
                alter_statement = text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type};")
                with engine.begin() as conn:
                    conn.execute(alter_statement)
                    logger.info(f"Added column {column_name} to table {table_model.__tablename__}")
