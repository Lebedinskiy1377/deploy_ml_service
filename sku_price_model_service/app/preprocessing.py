"""Turn an uploaded CSV into model features.

Mirrors load_training_data() in application/src/models/train_model.py: calendar
features come from `dates`, everything else from the reference tables.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import CATEGORICAL_FEATURES, FEATURES, REQUIRED_INPUT_COLUMNS
from .db import ReferenceData

SKU_ATTRIBUTES = (*CATEGORICAL_FEATURES, "creation_date", "expiration_date")
NO_DISCOUNT = 1.0


def _preview(values: Iterable[object], limit: int = 10) -> str:
    unique = sorted({str(value) for value in values})
    shown = ", ".join(unique[:limit])
    return shown if len(unique) <= limit else f"{shown} and {len(unique) - limit} more"


def _parse_sku(values: pd.Series) -> pd.Series:
    sku = pd.to_numeric(values, errors="coerce")
    invalid = sku.isna() | (sku != np.floor(sku))
    if invalid.any():
        raise ValueError(f"Column SKU must contain integer ids, got: {_preview(values[invalid])}")
    return sku.astype("int64")


def _parse_dates(values: pd.Series, column: str) -> pd.Series:
    # Parse text: integer dates such as 20191224 would otherwise be read as nanoseconds since 1970.
    dates = pd.to_datetime(values.astype(str), errors="coerce")
    invalid = dates.isna()
    if invalid.any():
        raise ValueError(f"Column {column} must contain dates, got: {_preview(values[invalid])}")
    return dates


def _parse_prices(values: pd.Series) -> pd.Series:
    prices = pd.to_numeric(values, errors="coerce")
    invalid = prices.isna() | (prices <= 0)
    if invalid.any():
        raise ValueError(f"Column price_per_sku must contain positive numbers, got: {_preview(values[invalid])}")
    return prices.astype("float64")


def _lookup(keys: pd.DataFrame, table: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    """Left join that keeps the row order and index of `keys`."""
    joined = keys.merge(table, on=on, how="left", validate="many_to_one")
    joined.index = keys.index
    return joined


def build_features(raw: pd.DataFrame, refs: ReferenceData) -> pd.DataFrame:
    """Return the model input for `raw`, indexed like `raw`, columns in FEATURES order.

    Raises ValueError with a user-facing message when the input cannot be scored.
    """
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if raw.empty:
        raise ValueError("The CSV file has no data rows")

    sku = _parse_sku(raw["SKU"])
    dates = _parse_dates(raw["dates"], "dates")
    price = _parse_prices(raw["price_per_sku"])

    unknown = set(sku) - set(refs.sku_dict["SKU"])
    if unknown:
        raise ValueError(f"Unknown SKU (missing from sku_dict): {_preview(unknown)}")

    features = pd.DataFrame(
        {
            "SKU": sku,
            "week_num": dates.dt.isocalendar().week.astype("int64"),
            "year": dates.dt.year.astype("int64"),
            "day": dates.dt.day.astype("int64"),
            "month": dates.dt.month.astype("int64"),
            "weekday": dates.dt.weekday.astype("int64"),
            "price": price,
        },
        index=raw.index,
    )

    # Promo calendar wins; a discount column in the CSV fills weeks without a promo.
    promo = _lookup(features[["SKU", "year", "week_num"]], refs.promo, on=["SKU", "year", "week_num"])
    discount = promo["discount"]
    if "discount" in raw.columns:
        discount = discount.fillna(pd.to_numeric(raw["discount"], errors="coerce"))
    features["discount"] = discount.fillna(NO_DISCOUNT).astype("float64")

    attributes = _lookup(features[["SKU"]], refs.sku_dict[["SKU", *SKU_ATTRIBUTES]], on=["SKU"])
    for column in CATEGORICAL_FEATURES:
        features[column] = attributes[column].astype("category")

    for prefix, column in (("creation", "creation_date"), ("expiration", "expiration_date")):
        lifecycle_date = _parse_dates(attributes[column], f"sku_dict.{column}")
        features[f"week_num_{prefix}"] = lifecycle_date.dt.isocalendar().week.astype("int64")
        features[f"year_{prefix}"] = lifecycle_date.dt.year.astype("int64")

    return features[list(FEATURES)]
