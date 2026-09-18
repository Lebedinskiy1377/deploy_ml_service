"""Test doubles and reference tables shared by the API tests."""

import numpy as np
import pandas as pd


class LinearDemandModel:
    """Stand-in for the MLflow model: demand = intercept - slope * price."""

    def __init__(self, intercept: float = 200.0, slope: float = 1.0) -> None:
        self.intercept = intercept
        self.slope = slope
        self.calls: list[pd.DataFrame] = []

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.calls.append(features)
        return self.intercept - self.slope * features["price"].to_numpy(dtype=float)


def reference_frames() -> dict[str, pd.DataFrame]:
    """Tables in the shape produced by scripts/seed_demo_db.py."""
    return {
        # 2019-12-24 falls on ISO week 52 of 2019.
        "promo": pd.DataFrame({"SKU": [101], "year": [2019], "week_num": [52], "discount": [0.8]}),
        "sku_dict": pd.DataFrame(
            {
                "sku_id": [101, 202],
                "fincode": ["15", "FE"],
                "ui1_code": ["151", "FE1"],
                "ui2_code": ["15100", "FE100"],
                "ui3_code": ["1510003", "FE10000"],
                "vendor": ["UEYMBB", "AWD3XQ"],
                "brand_code": ["WKXRWTP7", "79VL731U"],
                "creation_date": ["2018-01-17", "2018-03-01"],
                "expiration_date": ["2019-04-02 00:00:00", "2020-01-01 00:00:00"],
            }
        ),
        "prices": pd.DataFrame({"SKU": [101, 202], "price_per_sku": [100.0, 50.0], "cost": [80.0, 30.0]}),
    }
