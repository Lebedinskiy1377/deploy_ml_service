import numpy as np
import pandas as pd
import pytest

from app.preprocessing import build_features
from app.price_optimizer import OptimizationSettings, optimize_prices, price_grid

NO_MARGIN_PENALTY = OptimizationSettings(margin_penalty=0.0)


def features_with_price(reference_data, price: float) -> pd.DataFrame:
    request = pd.DataFrame({"dates": ["2019-06-01"], "SKU": [202], "price_per_sku": [price]})
    return build_features(request, reference_data)


def test_without_margin_penalty_maximises_gmv(reference_data, predictor):
    # demand = 200 - price, so GMV peaks at price 100.
    features = features_with_price(reference_data, 90.0)
    costs = pd.Series([30.0], index=features.index)

    result = optimize_prices(features, costs, predictor, NO_MARGIN_PENALTY)

    step = np.diff(price_grid(np.array([90.0]), NO_MARGIN_PENALTY)[0])[0]
    optimal = result["optimal_price"].iloc[0]
    assert abs(optimal - 100.0) <= step / 2
    assert result["expected_demand"].iloc[0] == pytest.approx(200.0 - optimal)
    assert result["gmv"].iloc[0] == pytest.approx(optimal * (200.0 - optimal))


def test_margin_penalty_moves_price_up_when_margin_is_below_target(reference_data, predictor):
    features = features_with_price(reference_data, 90.0)
    costs = pd.Series([80.0], index=features.index)  # margin ~11% at the base price

    without_penalty = optimize_prices(features, costs, predictor, NO_MARGIN_PENALTY)
    with_penalty = optimize_prices(features, costs, predictor, OptimizationSettings(margin_penalty=2.0))

    assert with_penalty["optimal_price"].iloc[0] > without_penalty["optimal_price"].iloc[0]


def test_base_scenario_uses_the_requested_price(request_frame, reference_data, predictor):
    features = build_features(request_frame, reference_data)
    costs = reference_data.costs_for(features["SKU"])

    result = optimize_prices(features, costs, predictor)

    assert list(result.index) == list(features.index)
    np.testing.assert_allclose(result["price_per_sku"], [99.5, 45.0])
    np.testing.assert_allclose(result["base_demand"], [200.0 - 99.5, 200.0 - 45.0])
    np.testing.assert_allclose(result["base_gmv"], result["price_per_sku"] * result["base_demand"])
    np.testing.assert_allclose(result["base_margin"], [(99.5 - 80.0) / 99.5, (45.0 - 30.0) / 45.0])


def test_optimal_price_stays_within_the_allowed_range(request_frame, reference_data, predictor):
    features = build_features(request_frame, reference_data)
    costs = reference_data.costs_for(features["SKU"])

    result = optimize_prices(features, costs, predictor)

    ratio = result["optimal_price"] / result["price_per_sku"]
    assert ratio.between(0.7 - 1e-9, 1.3 + 1e-9).all()


@pytest.mark.parametrize("settings", [{"candidate_count": 1}, {"price_change_limit": 1.0}])
def test_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        OptimizationSettings(**settings)
