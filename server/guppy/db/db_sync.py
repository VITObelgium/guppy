import logging

from guppy.db.models import LayerMetadata, TileStatistics

logger = logging.getLogger(__name__)
from sqlalchemy import MetaData, text
from guppy.db.db_session import engine


def keep_db_tables_in_sync():
    metadata = MetaData()
    metadata.reflect(bind=engine)
    for table_model in [LayerMetadata, TileStatistics]:
        table = metadata.tables[table_model.__tablename__]

        for column in table_model.__table__.columns:
            print(column)
            if column.name not in [c.name for c in table.columns]:
                column_type = column.type.compile(dialect=engine.dialect)
                column_name = column.name
                alter_statement = text(f"ALTER TABLE {table_model.__tablename__} ADD COLUMN {column_name} {column_type};")
                with engine.connect() as conn:
                    conn.execute(alter_statement)
                    logger.info(f"Added column {column_name} to table {table_model.__tablename__}")
