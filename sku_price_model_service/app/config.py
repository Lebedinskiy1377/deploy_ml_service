"""Service settings and the feature contract shared with the training pipeline.

FEATURES and CATEGORICAL_FEATURES must match application/src/models/train_model.py;
a test in the training suite checks that they do.
"""

import os
from typing import Final

MLFLOW_TRACKING_URI: Final[str] = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME: Final[str] = os.getenv("MODEL_NAME", "lgb_for_inference")
MODEL_ALIAS: Final[str] = os.getenv("MODEL_ALIAS", "champion")

REQUIRED_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "dates",
    "SKU",
    "price_per_sku",
)

CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "fincode",
    "ui1_code",
    "ui2_code",
    "ui3_code",
    "vendor",
    "brand_code",
)

FEATURES: Final[tuple[str, ...]] = (
    "SKU",
    "week_num",
    "year",
    "discount",
    "fincode",
    "ui1_code",
    "ui2_code",
    "ui3_code",
    "vendor",
    "brand_code",
    "week_num_expiration",
    "year_expiration",
    "week_num_creation",
    "year_creation",
    "day",
    "month",
    "weekday",
    "price",
)

DEFAULT_PRICE_CHANGE_LIMIT: Final[float] = 0.30
DEFAULT_PRICE_CANDIDATE_COUNT: Final[int] = 30
DEFAULT_MARGIN_PENALTY: Final[float] = 0.5
DEFAULT_TARGET_MARGIN: Final[float] = 0.5
