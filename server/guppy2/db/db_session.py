# /books/database/db_session.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from guppy2.config import config

db_url = 'postgresql+psycopg2://'
db_url += config.database.user
db_url += ':'
db_url += config.database.passwd
db_url += '@'
db_url += config.database.host
db_url += '/'
db_url += config.database.db

engine = create_engine(db_url, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
