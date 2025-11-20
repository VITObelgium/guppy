from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from guppy.config import config

if config.database.type == "postgres":
    db_url = "postgresql+psycopg2://"
    db_url += config.database.user
    db_url += ":"
    db_url += config.database.passwd
    db_url += "@"
    db_url += config.database.host
    db_url += "/"
    db_url += config.database.db
    connect_args = {}
    engine = create_engine(db_url, echo=False, connect_args=connect_args)

elif config.database.type == "sqlite":
    db_url = f"sqlite:///{config.database.db}"
    connect_args = {"check_same_thread": False}
    engine = create_engine(
        db_url,
        echo=False,
        connect_args=connect_args,
        poolclass=StaticPool,
        pool_pre_ping=True,
    )
else:
    raise Exception(
        f"Unknown database type {config.database.type}. supported types are 'postgres' and 'sqlite'"
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
