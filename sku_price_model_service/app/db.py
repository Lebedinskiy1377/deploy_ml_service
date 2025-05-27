import os
from sqlalchemy import create_engine

username = os.getenv("POSTGRES_USER", "root")
password = os.getenv("POSTGRES_PASSWORD", "root")
host = "pg_container"
port = 5432
database = os.getenv("POSTGRES_DB", "test_db")


engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{database}')