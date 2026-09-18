"""Pick the price that maximises GMV with a penalty for margins below target.

For every price candidate:
score = price * demand * (1 - margin_penalty * max(0, target_margin - margin)),
where margin = (price - cost) / price.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_MARGIN_PENALTY,
    DEFAULT_PRICE_CANDIDATE_COUNT,
    DEFAULT_PRICE_CHANGE_LIMIT,
    DEFAULT_TARGET_MARGIN,
)
from .demand_predictor import DemandPredictor


@dataclass(frozen=True)
class OptimizationSettings:
    price_change_limit: float = DEFAULT_PRICE_CHANGE_LIMIT  # candidates within +-30% of the base price
    candidate_count: int = DEFAULT_PRICE_CANDIDATE_COUNT
    margin_penalty: float = DEFAULT_MARGIN_PENALTY  # lambda in the score formula
    target_margin: float = DEFAULT_TARGET_MARGIN

    def __post_init__(self) -> None:
        if not 0 <= self.price_change_limit < 1:
            raise ValueError("price_change_limit must be in [0, 1)")
        if self.candidate_count < 2:
            raise ValueError("candidate_count must be at least 2")


def margin_share(price: np.ndarray, cost: np.ndarray) -> np.ndarray:
    price, cost = np.broadcast_arrays(np.asarray(price, dtype=float), np.asarray(cost, dtype=float))
    return np.divide(price - cost, price, out=np.zeros_like(price), where=price > 0)


def penalized_gmv(
    price: np.ndarray,
    demand: np.ndarray,
    cost: np.ndarray,
    settings: OptimizationSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (score, margin) for every price candidate."""
    margin = margin_share(price, cost)
    penalty = settings.margin_penalty * np.maximum(0.0, settings.target_margin - margin)
    return price * demand * (1.0 - penalty), margin


def price_grid(base_price: np.ndarray, settings: OptimizationSettings) -> np.ndarray:
    """Candidates per row: an even grid within +-price_change_limit plus the base price itself."""
    multipliers = np.union1d(
        np.linspace(1 - settings.price_change_limit, 1 + settings.price_change_limit, settings.candidate_count),
        [1.0],  # keeping the current price must always be an option
    )
    return np.asarray(base_price, dtype=float)[:, None] * multipliers[None, :]


def optimize_prices(
    features: pd.DataFrame,
    costs: pd.Series,
    predictor: DemandPredictor,
    settings: OptimizationSettings | None = None,
) -> pd.DataFrame:
    """Best price per row of `features`; the base scenario is the price from the request."""
    settings = settings or OptimizationSettings()
    base_price = features["price"].to_numpy(dtype=float)
    cost = costs.to_numpy(dtype=float)

    grid = price_grid(base_price, settings)
    demand = predictor.demand_curves(features, grid)
    score, margin = penalized_gmv(grid, demand, cost[:, None], settings)

    rows = np.arange(len(features))
    best = np.argmax(score, axis=1)
    optimal_price = grid[rows, best]
    expected_demand = demand[rows, best]
    base_demand = predictor.predict(features)

    return pd.DataFrame(
        {
            "SKU": features["SKU"].to_numpy(),
            "price_per_sku": base_price,
            "cost": cost,
            "base_demand": base_demand,
            "base_gmv": base_price * base_demand,
            "base_margin": margin_share(base_price, cost),
            "optimal_price": optimal_price,
            "expected_demand": expected_demand,
            "gmv": optimal_price * expected_demand,
            "margin": margin[rows, best],
            "score": score[rows, best],
        },
        index=features.index,
    )
