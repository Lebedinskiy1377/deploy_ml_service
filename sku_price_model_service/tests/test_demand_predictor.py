import numpy as np
import pandas as pd

from app.demand_predictor import DemandPredictor


class ArrayModel:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.calls = 0

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.calls += 1
        return self.output[: len(features)]


def test_predict_clips_negative_demand():
    predictor = DemandPredictor(ArrayModel(np.array([-2.0, 0.0, 3.5])))

    np.testing.assert_array_equal(predictor.predict(pd.DataFrame({"price": [1, 2, 3]})), [0.0, 0.0, 3.5])


def test_demand_curves_score_all_candidates_in_one_call_and_never_rise_with_price():
    # Raw model output rises between the 2nd and 3rd candidate of the first row.
    model = ArrayModel(np.array([10.0, 8.0, 9.0, 5.0, 4.0, 3.0]))
    predictor = DemandPredictor(model)
    features = pd.DataFrame({"SKU": [1, 2], "price": [10.0, 20.0]})
    grid = np.array([[9.0, 10.0, 11.0], [18.0, 20.0, 22.0]])

    curves = predictor.demand_curves(features, grid)

    assert model.calls == 1
    assert curves.shape == (2, 3)
    assert np.all(np.diff(curves, axis=1) <= 1e-12)
    np.testing.assert_allclose(curves[0], [10.0, 8.5, 8.5])
    np.testing.assert_allclose(curves[1], [5.0, 4.0, 3.0])
