"""End-to-end smoke test: train on the real dataset and register the model in a local MLflow store."""

import mlflow
import numpy as np
from mlflow import MlflowClient

from src.models.train_model import DEFAULT_DATA_PATH, FEATURES, load_training_data, train


def test_train_registers_a_champion_that_respects_price_monotonicity(tmp_path, monkeypatch):
    tracking_uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    result = train(
        n_trials=1,
        timeout=None,
        cv_splits=2,
        max_estimators=40,
        early_stopping_rounds=5,
        model_output=tmp_path / "lgb_model.txt",
    )

    assert result["model_version"] == "1"
    assert set(result["metrics"]) == {"MAE", "RMSE", "MAPE", "SMAPE", "WAPE", "r2_score"}
    client = MlflowClient(tracking_uri=tracking_uri)
    assert str(client.get_model_version_by_alias("lgb_for_inference", "champion").version) == "1"

    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model("models:/lgb_for_inference@champion")
    row = load_training_data(DEFAULT_DATA_PATH)[list(FEATURES)].iloc[[-1]]
    candidates = row.iloc[np.zeros(20, dtype=int)].reset_index(drop=True)
    candidates["price"] = np.linspace(0.5, 1.5, 20) * row["price"].iloc[0]

    demand = model.predict(candidates)

    assert np.all(np.diff(demand) <= 1e-9), "demand must not grow with price"
