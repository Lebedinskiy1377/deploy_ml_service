from typing import Final

MODEL_NAME: Final[str] = "lgb_for_inference"
MODEL_STAGE: Final[str] = "Staging"

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
