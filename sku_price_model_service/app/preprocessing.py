import numpy as np
import pandas as pd

from config import CATEGORICAL_FEATURES, FEATURES, REQUIRED_INPUT_COLUMNS
from db import engine

SKU_DICT_COLUMNS = [
    'fincode',
    'ui1_code',
    'ui2_code',
    'ui3_code',
    'vendor',
    'brand_code',
    'creation_date',
    'expiration_date',
]


def _validate_input_columns(data: pd.DataFrame) -> None:
    missing_columns = sorted(set(REQUIRED_INPUT_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def _prefer_lookup_columns(data: pd.DataFrame, columns: list[str], suffix: str) -> pd.DataFrame:
    for column in columns:
        lookup_column = f"{column}{suffix}"
        if lookup_column in data.columns:
            data[column] = data[lookup_column].combine_first(data[column])

    return data.drop(
        columns=[f"{column}{suffix}" for column in columns if f"{column}{suffix}" in data.columns],
    )


def preprocessing(raw_data: pd.DataFrame) -> pd.DataFrame:
    data: pd.DataFrame = raw_data.copy()
    promo: pd.DataFrame = pd.read_sql_query("SELECT * FROM promo", engine)
    sku_dict: pd.DataFrame = pd.read_sql_query("SELECT * FROM sku_dict", engine)
    prices: pd.DataFrame = pd.read_sql_query("SELECT * FROM prices", engine)[['SKU', 'cost']]

    _validate_input_columns(data)
    data = data.rename(columns={'price_per_sku': 'price'})

    dates = pd.to_datetime(data["dates"], errors="raise")
    data["week_num"] = dates.dt.isocalendar().week
    data["year"] = dates.dt.year
    data["day"] = dates.dt.day
    data["month"] = dates.dt.month
    data["weekday"] = dates.dt.weekday

    data = pd.merge(data, promo, on=['SKU', 'year', 'week_num'], how='left', suffixes=('', '_promo'))
    if 'discount_promo' in data.columns:
        data['discount'] = data['discount_promo'].combine_first(data['discount'])
        data = data.drop(columns=['discount_promo'])
    data.discount = data.discount.fillna(1.)
    data.dates = dates
    sku_dict = sku_dict.rename(columns={'sku_id': 'SKU'})

    data = pd.merge(
        data,
        sku_dict,
        on='SKU',
        how='left',
        suffixes=('', '_dict'),
    )
    data = _prefer_lookup_columns(data, SKU_DICT_COLUMNS, '_dict')

    data = pd.merge(data, prices, on='SKU', how='left', suffixes=('', '_lookup'))
    if 'cost_lookup' in data.columns:
        data['cost'] = data['cost_lookup'].combine_first(data['cost'])
        data = data.drop(columns=['cost_lookup'])

    data['price'] = pd.to_numeric(data['price'], errors='coerce').fillna(0.0)
    data['cost'] = pd.to_numeric(data['cost'], errors='coerce').fillna(0.0)

    data['margin'] = np.where(
        data['price'] > 0,
        (data['price'] - data['cost']) / data['price'],
        0.0,
    )

    data['week_num_expiration'] = pd.to_datetime(data.expiration_date).dt.isocalendar().week
    data['year_expiration'] = pd.to_datetime(data.expiration_date).dt.year
    data['week_num_creation'] = pd.to_datetime(data.creation_date).dt.isocalendar().week
    data['year_creation'] = pd.to_datetime(data.creation_date).dt.year

    X: pd.DataFrame = data.drop(['dates', 'creation_date', 'expiration_date'], axis=1)

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype('category')

    val_data: pd.DataFrame = X[list(FEATURES)]

    return val_data
