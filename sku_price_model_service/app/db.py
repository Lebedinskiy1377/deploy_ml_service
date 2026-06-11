import os

from sqlalchemy import create_engine

username = os.getenv("POSTGRES_USER", "root")
password = os.getenv("POSTGRES_PASSWORD", "root")
host = os.getenv("POSTGRES_HOST", "pg_container")
port = int(os.getenv("POSTGRES_PORT", "5432"))
database = os.getenv("POSTGRES_DB", "test_db")

engine = create_engine(
    f"postgresql://{username}:{password}@{host}:{port}/{database}",
    pool_pre_ping=True,
)
