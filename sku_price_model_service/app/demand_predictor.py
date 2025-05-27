import mlflow
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from typing import Any


class DemandPredictor:
    def __init__(self, model_name: str = "lgb_for_inference", model_stage: str = "Staging") -> None:
        self.model: Any = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")
        self.isotonic_reg: IsotonicRegression = IsotonicRegression(increasing=False)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        demand: np.ndarray = self.model.predict(data)
        return demand

    def adjust_demand_with_price(self, data: pd.DataFrame, price_candidates: np.ndarray) -> np.ndarray:
        base_price = data['price'].values[0]
        price_elasticity = 1.2

        demands = []
        for price in price_candidates:
            data_copy = data.copy()
            data_copy['price'] = price
            model_demand = self.model.predict(data_copy)[0]

            price_ratio = price / base_price
            elasticity_correction = price_ratio ** (-price_elasticity)
            adjusted_demand = model_demand * elasticity_correction

            demands.append(model_demand) #adjusted_demand)

        demands = np.array(demands)
        price_candidates = np.array(price_candidates).reshape(-1, 1)  # Формат для IsotonicRegression
        self.isotonic_reg.fit(price_candidates, demands)
        adjusted_demands = self.isotonic_reg.predict(price_candidates)


        return adjusted_demands # np.array(demands)
