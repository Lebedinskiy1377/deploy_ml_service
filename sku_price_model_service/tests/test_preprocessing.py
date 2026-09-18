import pandas as pd
import pytest

from app.config import CATEGORICAL_FEATURES, FEATURES
from app.preprocessing import build_features


def test_builds_features_in_training_order(request_frame, reference_data):
    features = build_features(request_frame, reference_data)

    assert list(features.columns) == list(FEATURES)
    first = features.iloc[0]
    assert (first["SKU"], first["year"], first["week_num"]) == (101, 2019, 52)
    assert (first["day"], first["month"], first["weekday"]) == (24, 12, 1)
    assert (first["year_creation"], first["week_num_creation"]) == (2018, 3)
    assert (first["year_expiration"], first["week_num_expiration"]) == (2019, 14)
    assert first["price"] == 99.5
    assert first["fincode"] == "15"


def test_uses_training_dtypes(request_frame, reference_data):
    features = build_features(request_frame, reference_data)

    for column in CATEGORICAL_FEATURES:
        assert isinstance(features[column].dtype, pd.CategoricalDtype), column
    integer_columns = [c for c in FEATURES if c not in CATEGORICAL_FEATURES and c not in ("price", "discount")]
    for column in integer_columns:
        assert features[column].dtype == "int64", column
    assert features["price"].dtype == features["discount"].dtype == "float64"


def test_keeps_the_index_of_the_request(request_frame, reference_data):
    request_frame.index = [10, 3]

    features = build_features(request_frame, reference_data)

    assert list(features.index) == [10, 3]
    assert list(features["SKU"]) == [101, 202]


def test_discount_comes_from_promo_calendar_then_from_the_request(request_frame, reference_data):
    assert list(build_features(request_frame, reference_data)["discount"]) == [0.8, 1.0]

    request_frame["discount"] = [0.5, 0.9]

    # The promo calendar wins for SKU 101; SKU 202 has no promo that week.
    assert list(build_features(request_frame, reference_data)["discount"]) == [0.8, 0.9]


def test_rejects_unknown_sku(request_frame, reference_data):
    request_frame.loc[1, "SKU"] = 999

    with pytest.raises(ValueError, match="Unknown SKU.*999"):
        build_features(request_frame, reference_data)


@pytest.mark.filterwarnings("ignore:Could not infer format")
@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("dates", "yesterday", "dates must contain dates"),
        ("price_per_sku", -1, "positive numbers"),
        ("price_per_sku", "free", "positive numbers"),
        ("SKU", "abc", "integer ids"),
        ("SKU", 101.5, "integer ids"),
    ],
)
def test_rejects_invalid_values(request_frame, reference_data, column, value, message):
    request_frame[column] = request_frame[column].astype(object)
    request_frame.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        build_features(request_frame, reference_data)


def test_rejects_missing_columns(request_frame, reference_data):
    with pytest.raises(ValueError, match="Missing required columns: price_per_sku"):
        build_features(request_frame.drop(columns="price_per_sku"), reference_data)


def test_rejects_empty_request(request_frame, reference_data):
    with pytest.raises(ValueError, match="no data rows"):
        build_features(request_frame.iloc[0:0], reference_data)
