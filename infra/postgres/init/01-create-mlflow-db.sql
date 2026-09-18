-- Runs once, when the PostgreSQL volume is initialised.
-- MLflow keeps its metadata in a separate database; the pricing tables
-- (promo, sku_dict, prices) stay in POSTGRES_DB.
CREATE DATABASE mlflow;
