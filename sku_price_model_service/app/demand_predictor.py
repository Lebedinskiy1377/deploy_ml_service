"""Demand model loaded from the MLflow model registry."""

from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.isotonic import IsotonicRegression

from .config import MLFLOW_TRACKING_URI, MODEL_ALIAS, MODEL_NAME


class DemandPredictor:
    def __init__(self, model: Any, version: str | None = None) -> None:
        self._model = model
        self.version = version

    @classmethod
    def from_registry(
        cls,
        name: str = MODEL_NAME,
        alias: str = MODEL_ALIAS,
        tracking_uri: str = MLFLOW_TRACKING_URI,
    ) -> "DemandPredictor":
        mlflow.set_tracking_uri(tracking_uri)
        model_version = MlflowClient().get_model_version_by_alias(name, alias)
        model = mlflow.pyfunc.load_model(f"models:/{name}/{model_version.version}")
        return cls(model, version=str(model_version.version))

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Expected demand per row. Negative regression outputs are clipped to zero."""
        return np.clip(np.asarray(self._model.predict(features), dtype=float), 0.0, None)

    def demand_curves(self, features: pd.DataFrame, price_grid: np.ndarray) -> np.ndarray:
        """Demand of every row at every price of its grid, shape (n_rows, n_prices).

        All candidates are scored in a single model call. Each curve is made
        non-increasing in price with isotonic regression: the model already has a
        monotone price constraint, so this only guards against models without it.
        """
        n_rows, n_prices = price_grid.shape
        candidates = features.iloc[np.repeat(np.arange(n_rows), n_prices)].reset_index(drop=True)
        candidates["price"] = price_grid.reshape(-1)
        demand = self.predict(candidates).reshape(n_rows, n_prices)

        isotonic = IsotonicRegression(increasing=False)
        curves = [isotonic.fit_transform(prices, curve) for prices, curve in zip(price_grid, demand, strict=True)]
        return np.vstack(curves)
