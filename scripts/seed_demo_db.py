"""Load the demo reference tables (promo, sku_dict, prices) into PostgreSQL."""

import argparse
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import URL, Engine, create_engine

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "application" / "data" / "processed" / "sku_sales.csv"

SKU_DICT_COLUMNS = [
    "SKU",
    "fincode",
    "ui1_code",
    "ui2_code",
    "ui3_code",
    "vendor",
    "brand_code",
    "creation_date",
    "expiration_date",
]


def get_engine() -> Engine:
    # Defaults match docker-compose.yml when the script runs on the host.
    url = URL.create(
        "postgresql+psycopg2",
        username=os.getenv("POSTGRES_USER", "pricing"),
        password=os.getenv("POSTGRES_PASSWORD", "pricing"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        database=os.getenv("POSTGRES_DB", "pricing"),
    )
    return create_engine(url)


def build_prices(data: pd.DataFrame) -> pd.DataFrame:
    price_column = "price_per_sku" if "price_per_sku" in data.columns else "price"
    latest_prices = data.sort_values("dates").groupby("SKU", as_index=False).tail(1).copy()
    latest_prices = latest_prices.rename(columns={price_column: "price_per_sku"})

    if "margin" in latest_prices.columns:
        inferred_cost = latest_prices["price_per_sku"] - latest_prices["margin"]
        latest_prices["cost"] = inferred_cost.where(
            (inferred_cost > 0) & (inferred_cost < latest_prices["price_per_sku"]),
            latest_prices["price_per_sku"] * 0.7,
        )
    else:
        latest_prices["cost"] = latest_prices["price_per_sku"] * 0.7

    return latest_prices[["SKU", "price_per_sku", "cost"]].drop_duplicates("SKU")


def build_reference_tables(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Derive the promo calendar, SKU dictionary and current prices from sales history."""
    # Only real promotions: for weeks without one the API falls back to the discount
    # sent with the request, or to 1.0 (no discount).
    promo = data.loc[data["discount"] != 1.0, ["SKU", "year", "week_num", "discount"]].drop_duplicates()
    sku_dict = data[SKU_DICT_COLUMNS].drop_duplicates("SKU").rename(columns={"SKU": "sku_id"})

    return {"promo": promo, "sku_dict": sku_dict, "prices": build_prices(data)}


def seed_database(data_path: str, if_exists: str, engine: Engine | None = None) -> None:
    tables = build_reference_tables(pd.read_csv(data_path))
    engine = engine or get_engine()

    for name, table in tables.items():
        table.to_sql(name, engine, if_exists=if_exists, index=False)
        print(f"Seeded {name}: {len(table)} rows")


def parse_args():
    parser = argparse.ArgumentParser(description="Seed demo PostgreSQL tables for the pricing service.")
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="Path to the source CSV with historical SKU observations.",
    )
    parser.add_argument(
        "--if-exists",
        choices=("fail", "replace", "append"),
        default="replace",
        help="Behavior when target tables already exist.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_database(data_path=args.data_path, if_exists=args.if_exists)
