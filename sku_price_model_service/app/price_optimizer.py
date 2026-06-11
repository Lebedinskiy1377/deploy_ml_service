import logging
from typing import Any

import numpy as np
import pandas as pd

from config import (
    CATEGORICAL_FEATURES,
    DEFAULT_MARGIN_PENALTY,
    DEFAULT_PRICE_CANDIDATE_COUNT,
    DEFAULT_PRICE_CHANGE_LIMIT,
    DEFAULT_TARGET_MARGIN,
    FEATURES,
)
from db import engine
from demand_predictor import DemandPredictor

logger = logging.getLogger(__name__)


def _prepare_model_row(row: pd.Series) -> pd.DataFrame:
    row_data = pd.DataFrame([{feature: row[feature] for feature in FEATURES}])

    row_data["SKU"] = row_data["SKU"].astype(np.int64)
    row_data["week_num"] = row_data["week_num"].astype(np.uint32)
    row_data["year"] = row_data["year"].astype(np.int32)
    row_data["discount"] = row_data["discount"].astype(np.float64)
    row_data["week_num_expiration"] = row_data["week_num_expiration"].astype(np.uint32)
    row_data["year_expiration"] = row_data["year_expiration"].astype(np.int32)
    row_data["week_num_creation"] = row_data["week_num_creation"].astype(np.uint32)
    row_data["year_creation"] = row_data["year_creation"].astype(np.int32)
    row_data["day"] = row_data["day"].astype(np.int32)
    row_data["month"] = row_data["month"].astype(np.int32)
    row_data["weekday"] = row_data["weekday"].astype(np.int32)
    row_data["price"] = row_data["price"].astype(np.float64)

    for column in CATEGORICAL_FEATURES:
        row_data[column] = row_data[column].astype("category")

    return row_data


def _score_candidate(
    price: float,
    demand: float,
    cost: float,
    lambda_param: float,
    target_margin: float,
) -> tuple[float, float]:
    margin = (price - cost) / price if price > 0 else 0.0
    penalty = lambda_param * max(0.0, target_margin - margin)
    score = price * demand * (1 - penalty)

    return score, margin


def optimize_price(
    data: pd.DataFrame,
    predictor: DemandPredictor,
    lambda_param: float = DEFAULT_MARGIN_PENALTY,
    target_margin: float = DEFAULT_TARGET_MARGIN,
    price_change_limit: float = DEFAULT_PRICE_CHANGE_LIMIT,
    price_candidate_count: int = DEFAULT_PRICE_CANDIDATE_COUNT,
) -> pd.DataFrame:
    """
    Optimizes each SKU price by maximizing a GMV-based score with a margin penalty.
    """
    prices: pd.DataFrame = pd.read_sql_query(
        "SELECT SKU, price_per_sku, cost FROM prices",
        engine,
    )
    data = pd.merge(data, prices, on="SKU", how="left")

    if data[["price_per_sku", "cost"]].isna().any(axis=None):
        missing_skus = data.loc[
            data[["price_per_sku", "cost"]].isna().any(axis=1),
            "SKU",
        ].unique()
        raise ValueError(f"Missing price or cost data for SKU: {', '.join(map(str, missing_skus))}")

    data["base_demand"] = predictor.predict(data[list(FEATURES)])

    results: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        sku = int(row["SKU"])
        base_price = float(row["price"])
        cost = float(row["cost"])

        if base_price <= 0:
            raise ValueError(f"Base price must be greater than zero for SKU {sku}")

        price_candidates = np.linspace(
            base_price * (1 - price_change_limit),
            base_price * (1 + price_change_limit),
            price_candidate_count,
        )

        row_data = _prepare_model_row(row)
        demands = predictor.adjust_demand_with_price(row_data, price_candidates)
        demands = np.nan_to_num(demands, nan=0.0, posinf=0.0, neginf=0.0)
        logger.debug("SKU %s candidate prices=%s demands=%s", sku, price_candidates, demands)

        scores = np.zeros_like(price_candidates)
        margins = np.zeros_like(price_candidates)
        for idx, (price, demand) in enumerate(zip(price_candidates, demands)):
            scores[idx], margins[idx] = _score_candidate(
                price=float(price),
                demand=float(demand),
                cost=cost,
                lambda_param=lambda_param,
                target_margin=target_margin,
            )

        best_idx = int(np.argmax(scores))
        best_price = float(price_candidates[best_idx])
        best_demand = float(demands[best_idx])

        results.append(
            {
                "SKU": sku,
                "optimal_price": best_price,
                "expected_demand": best_demand,
                "gmv": best_price * best_demand,
                "margin": float(margins[best_idx]),
                "score": float(scores[best_idx]),
                "base_demand": float(row["base_demand"]),
            }
        )

    return pd.DataFrame(results).merge(prices, on="SKU", how="left")
