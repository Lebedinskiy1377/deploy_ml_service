import importlib.util
import unittest
from pathlib import Path

import numpy as np

from src.models.metrics import mape, smape, wape
from src.models.train_model import (
    DEFAULT_DATA_PATH,
    FEATURES,
    build_model_params,
    load_training_data,
    temporal_holdout_split,
)


class TrainingPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_training_data(DEFAULT_DATA_PATH)

    def test_dataset_matches_inference_contract(self):
        self.assertEqual(list(self.data.columns), ["dates", *FEATURES, "num_purchases"])
        self.assertNotIn("margin", self.data.columns)
        self.assertFalse(self.data.isna().any(axis=None))

    def test_temporal_holdout_has_no_date_overlap(self):
        train, holdout, cutoff = temporal_holdout_split(self.data, 0.9)

        self.assertLess(train["dates"].max(), cutoff)
        self.assertGreaterEqual(holdout["dates"].min(), cutoff)
        self.assertTrue(set(train["dates"]).isdisjoint(set(holdout["dates"])))

    def test_price_monotonicity_is_negative(self):
        params = build_model_params(None, list(FEATURES))
        price_index = list(FEATURES).index("price")

        self.assertEqual(params["objective"], "regression")
        self.assertEqual(params["monotone_constraints"][price_index], -1)
        self.assertEqual(sum(abs(value) for value in params["monotone_constraints"]), 1)

    def test_service_and_training_features_match(self):
        root = Path(__file__).resolve().parents[2]
        config_path = root / "sku_price_model_service" / "app" / "config.py"
        spec = importlib.util.spec_from_file_location("service_config", config_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        self.assertEqual(tuple(module.FEATURES), FEATURES)

    def test_percentage_metrics_are_finite_with_zeros(self):
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.0, 1.5, 1.0])

        self.assertTrue(np.isfinite(mape(y_true, y_pred)))
        self.assertTrue(np.isfinite(smape(y_true, y_pred)))
        self.assertTrue(np.isfinite(wape(y_true, y_pred)))


if __name__ == "__main__":
    unittest.main()
