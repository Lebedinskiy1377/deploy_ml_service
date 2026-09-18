"""PostgreSQL connection and loading of the reference tables."""

import os
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from sqlalchemy import URL, Engine, create_engine, text


def database_url() -> URL:
    # Defaults match docker-compose.yml when the API runs on the host.
    return URL.create(
        "postgresql+psycopg2",
        username=os.getenv("POSTGRES_USER", "pricing"),
        password=os.getenv("POSTGRES_PASSWORD", "pricing"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        database=os.getenv("POSTGRES_DB", "pricing"),
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(database_url(), pool_pre_ping=True)


@dataclass(frozen=True)
class ReferenceData:
    """Lookup tables that enrich an uploaded CSV with model features and costs."""

    promo: pd.DataFrame  # SKU, year, week_num, discount
    sku_dict: pd.DataFrame  # SKU, product hierarchy, vendor, brand, creation/expiration dates
    prices: pd.DataFrame  # SKU, price_per_sku, cost

    @classmethod
    def from_frames(
        cls,
        promo: pd.DataFrame,
        sku_dict: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> "ReferenceData":
        """Normalise raw tables: one row per lookup key and consistent key dtypes."""
        promo = promo[["SKU", "year", "week_num", "discount"]].astype(
            {"SKU": "int64", "year": "int64", "week_num": "int64", "discount": "float64"}
        )
        sku_dict = sku_dict.rename(columns={"sku_id": "SKU"}).astype({"SKU": "int64"})
        prices = prices[["SKU", "price_per_sku", "cost"]].astype(
            {"SKU": "int64", "price_per_sku": "float64", "cost": "float64"}
        )

        return cls(
            promo=promo.drop_duplicates(["SKU", "year", "week_num"], keep="last").reset_index(drop=True),
            sku_dict=sku_dict.drop_duplicates("SKU", keep="last").reset_index(drop=True),
            prices=prices.drop_duplicates("SKU", keep="last").reset_index(drop=True),
        )

    def costs_for(self, sku: pd.Series) -> pd.Series:
        """Unit cost for every SKU in `sku`, aligned with its index."""
        costs = sku.to_frame("SKU").merge(self.prices, on="SKU", how="left", validate="many_to_one")["cost"]
        costs.index = sku.index
        missing = sorted(set(sku[costs.isna()]))
        if missing:
            raise ValueError(f"No cost in the prices table for SKU: {', '.join(map(str, missing))}")
        return costs


def load_reference_data(engine: Engine) -> ReferenceData:
    with engine.connect() as connection:
        promo = pd.read_sql_query(text('SELECT "SKU", year, week_num, discount FROM promo'), connection)
        sku_dict = pd.read_sql_query(text("SELECT * FROM sku_dict"), connection)
        prices = pd.read_sql_query(text('SELECT "SKU", price_per_sku, cost FROM prices'), connection)

    return ReferenceData.from_frames(promo=promo, sku_dict=sku_dict, prices=prices)
