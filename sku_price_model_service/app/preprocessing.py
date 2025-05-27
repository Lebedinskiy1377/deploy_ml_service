import pandas as pd
from db import engine
from config import FEATURES


def preprocessing(raw_data: pd.DataFrame) -> pd.DataFrame:
    data: pd.DataFrame = raw_data.copy()
    promo: pd.DataFrame = pd.read_sql_query("SELECT * FROM promo", engine)
    sku_dict: pd.DataFrame = pd.read_sql_query("SELECT * FROM sku_dict", engine)
    prices: pd.DataFrame = pd.read_sql_query("SELECT * FROM prices", engine)[['SKU', 'cost']]

    data['week_num'] = pd.to_datetime(data.dates).dt.isocalendar().week
    data['year'] = pd.to_datetime(data.dates).dt.year
    data['day'] = pd.to_datetime(data.dates).dt.day
    data['month'] = pd.to_datetime(data.dates).dt.month
    data['weekday'] = pd.to_datetime(data.dates).dt.weekday

    data = pd.merge(data, promo, on=['SKU', 'year', 'week_num'], how='left')
    data.discount = data.discount.fillna(1.)
    data.dates = pd.to_datetime(data.dates)
    sku_dict = sku_dict.rename(columns={'sku_id': 'SKU'})

    data = pd.merge(
        data,
        sku_dict,
        on='SKU',
        how='left',
    )

    data = pd.merge(data, prices, on='SKU', how='left').rename(columns={'price_per_sku': 'price'})

    data['price'] = data['price'].fillna(0.0)
    data['cost'] = data['cost'].fillna(0.0)

    data['margin'] = (data['price'] - data['cost']) / data['price']
    data['margin'] = data['margin'].fillna(0.0)

    data['week_num_expiration'] = pd.to_datetime(data.expiration_date).dt.isocalendar().week
    data['year_expiration'] = pd.to_datetime(data.expiration_date).dt.year
    data['week_num_creation'] = pd.to_datetime(data.creation_date).dt.isocalendar().week
    data['year_creation'] = pd.to_datetime(data.creation_date).dt.year

    X: pd.DataFrame = data.drop(['dates', 'creation_date', 'expiration_date'], axis=1)

    cat_str_features: list[str] = ['ui1_code', 'ui2_code', 'ui3_code', 'vendor', 'brand_code', 'fincode']
    for col in cat_str_features:
        X[col] = X[col].astype('category')

    val_data: pd.DataFrame = X[FEATURES]

    return val_data
