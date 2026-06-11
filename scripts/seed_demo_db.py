import argparse
import os

import pandas as pd
from sqlalchemy import create_engine


def get_engine():
    username = os.getenv("POSTGRES_USER", "root")
    password = os.getenv("POSTGRES_PASSWORD", "root")
    host = os.getenv("POSTGRES_HOST", "db")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    database = os.getenv("POSTGRES_DB", "test_db")

    return create_engine(f"postgresql://{username}:{password}@{host}:{port}/{database}")


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


def seed_database(data_path: str, if_exists: str) -> None:
    data = pd.read_csv(data_path)

    promo = data[["SKU", "year", "week_num", "discount"]].drop_duplicates()
    sku_dict = (
        data[
            [
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
        ]
        .drop_duplicates("SKU")
        .rename(columns={"SKU": "sku_id"})
    )
    prices = build_prices(data)

    engine = get_engine()
    promo.to_sql("promo", engine, if_exists=if_exists, index=False)
    sku_dict.to_sql("sku_dict", engine, if_exists=if_exists, index=False)
    prices.to_sql("prices", engine, if_exists=if_exists, index=False)

    print(f"Seeded promo: {len(promo)} rows")
    print(f"Seeded sku_dict: {len(sku_dict)} rows")
    print(f"Seeded prices: {len(prices)} rows")


def parse_args():
    parser = argparse.ArgumentParser(description="Seed demo PostgreSQL tables for the pricing service.")
    parser.add_argument(
        "--data-path",
        default="/workspace/application/notebooks/data.csv",
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
