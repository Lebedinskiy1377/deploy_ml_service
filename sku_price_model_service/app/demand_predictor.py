import mlflow
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from typing import Any

from config import MODEL_ALIAS, MODEL_NAME


class DemandPredictor:
    def __init__(self, model_name: str = MODEL_NAME, model_alias: str = MODEL_ALIAS) -> None:
        self.model: Any = mlflow.pyfunc.load_model(f"models:/{model_name}@{model_alias}")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(data), dtype=float)

    def adjust_demand_with_price(self, data: pd.DataFrame, price_candidates: np.ndarray) -> np.ndarray:
        demands: list[float] = []
        for price in price_candidates:
            data_copy = data.copy()
            data_copy['price'] = price
            demands.append(float(self.model.predict(data_copy)[0]))

        demands = np.array(demands)
        isotonic_reg = IsotonicRegression(increasing=False, out_of_bounds="clip")
        isotonic_reg.fit(price_candidates, demands)

        return isotonic_reg.predict(price_candidates)
