import fire
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


def predict(test: str = "data/raw/test.csv",
            promo: str = "data/raw/promo_1510002.csv",
            sku_dict: str = "data/raw/sku_dict.csv"):
    """
    Predict price with sku and date
    """
    test = pd.read_csv(test)
    promo = pd.read_csv(promo)
    sku_dict = pd.read_csv(sku_dict)

    test['week_num'] = pd.to_datetime(test.dates).dt.isocalendar().week
    test['year'] = pd.to_datetime(test.dates).dt.year

    test = pd.merge(test, promo, on=['SKU', 'year', 'week_num'], how='left')
    test.discount = test.discount.fillna(1.)
    test.dates = pd.to_datetime(test.dates)

    test = pd.merge(
        test,
        sku_dict.rename(columns={'sku_id': 'SKU'}),
        on='SKU',
        how='left',
    )

    test['week_num_expiration'] = pd.to_datetime(test.expiration_date).dt.isocalendar().week
    test['year_expiration'] = pd.to_datetime(test.expiration_date).dt.year

    test['week_num_creation'] = pd.to_datetime(test.creation_date).dt.isocalendar().week
    test['year_creation'] = pd.to_datetime(test.creation_date).dt.year

    test['day'] = pd.to_datetime(test.dates).dt.day
    test['month'] = pd.to_datetime(test.dates).dt.month

    X = test.drop(['dates', 'creation_date', 'expiration_date'], axis=1)

    cat_str_features = ['fincode', 'ui1_code', 'ui2_code', 'ui3_code', 'vendor', 'brand_code']

    for col in cat_str_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        X[col] = X[col].astype('int64')

    dval = xgb.DMatrix(X)

    model = xgb.Booster(
        model_file="models/xgboost.json",
    )

    y_pred = model.predict(dval)

    test = test[['dates', 'SKU']]
    test['price_per_sku'] = y_pred
    test.to_csv("data/processed/submission.csv", index=False)
    print("Success!")


if __name__ == "__main__":
    fire.Fire(predict)
