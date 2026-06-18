from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterator

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from .metrics import mape, smape, wape

load_dotenv()

RANDOM_STATE = 42
DEFAULT_TRAIN_FRACTION = 0.9
DEFAULT_MODEL_NAME = "lgb_for_inference"
DEFAULT_MODEL_ALIAS = "champion"
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "notebooks" / "data.csv"
DEFAULT_MODEL_OUTPUT = Path(__file__).resolve().parents[2] / "models" / "lgb_model.txt"

CATEGORICAL_FEATURES = (
    "fincode",
    "ui1_code",
    "ui2_code",
    "ui3_code",
    "vendor",
    "brand_code",
)

FEATURES = (
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

SOURCE_COLUMNS = (
    "dates",
    "SKU",
    "price_per_sku",
    "num_purchases",
    "discount",
    "fincode",
    "ui1_code",
    "ui2_code",
    "ui3_code",
    "vendor",
    "brand_code",
    "creation_date",
    "expiration_date",
)


def load_training_data(
    train_path: str | Path = DEFAULT_DATA_PATH,
    target: str = "num_purchases",
) -> pd.DataFrame:
    path = Path(train_path)
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")

    data = pd.read_csv(path)
    required_columns = set(SOURCE_COLUMNS) | {target}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required training columns: {', '.join(missing_columns)}")

    if data.duplicated(["dates", "SKU"]).any():
        duplicate_count = int(data.duplicated(["dates", "SKU"]).sum())
        raise ValueError(f"Found {duplicate_count} duplicate dates/SKU rows")

    data["dates"] = pd.to_datetime(data["dates"], errors="raise")
    data["creation_date"] = pd.to_datetime(data["creation_date"], errors="raise")
    data["expiration_date"] = pd.to_datetime(data["expiration_date"], errors="raise")
    data = data.sort_values(["dates", "SKU"]).reset_index(drop=True)
    data = data.rename(columns={"price_per_sku": "price"})

    data["day"] = data["dates"].dt.day.astype("int64")
    data["month"] = data["dates"].dt.month.astype("int64")
    data["week_num"] = data["dates"].dt.isocalendar().week.astype("int64")
    data["year"] = data["dates"].dt.year.astype("int64")
    data["weekday"] = data["dates"].dt.weekday.astype("int64")
    data["week_num_expiration"] = data["expiration_date"].dt.isocalendar().week.astype("int64")
    data["year_expiration"] = data["expiration_date"].dt.year.astype("int64")
    data["week_num_creation"] = data["creation_date"].dt.isocalendar().week.astype("int64")
    data["year_creation"] = data["creation_date"].dt.year.astype("int64")

    model_columns = ["dates", *FEATURES, target]
    data = data[model_columns].copy()

    if data[model_columns].isna().any(axis=None):
        missing = data[model_columns].isna().sum()
        details = ", ".join(f"{column}={count}" for column, count in missing.items() if count)
        raise ValueError(f"Training data contains missing values: {details}")
    if (data["price"] <= 0).any():
        raise ValueError("Training data contains non-positive prices")
    if (data[target] < 0).any():
        raise ValueError(f"Training target {target} contains negative values")

    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].astype("category")

    return data


def temporal_holdout_split(
    data: pd.DataFrame,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    unique_dates = np.sort(data["dates"].unique())
    if len(unique_dates) < 3:
        raise ValueError("At least three unique dates are required for a temporal split")

    split_index = min(max(int(len(unique_dates) * train_fraction), 1), len(unique_dates) - 1)
    cutoff_date = pd.Timestamp(unique_dates[split_index])
    train = data[data["dates"] < cutoff_date].copy()
    holdout = data[data["dates"] >= cutoff_date].copy()

    return train, holdout, cutoff_date


def iter_date_splits(
    data: pd.DataFrame,
    n_splits: int,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    unique_dates = np.sort(data["dates"].unique())
    if len(unique_dates) <= n_splits:
        raise ValueError(f"Not enough unique dates for {n_splits} CV splits")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_date_indices, validation_date_indices in splitter.split(unique_dates):
        train_dates = unique_dates[train_date_indices]
        validation_dates = unique_dates[validation_date_indices]
        yield (
            data[data["dates"].isin(train_dates)],
            data[data["dates"].isin(validation_dates)],
        )


def calculate_price_elasticity(
    data: pd.DataFrame,
    target: str,
) -> float:
    positive = data[(data["price"] > 0) & (data[target] > 0)]
    if len(positive) < 2 or positive["price"].nunique() < 2:
        return float("nan")

    return float(np.polyfit(np.log(positive["price"]), np.log(positive[target]), 1)[0])


def regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    prediction = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    return {
        "MAE": float(mean_absolute_error(y_true, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "MAPE": float(mape(y_true.to_numpy(), prediction)),
        "SMAPE": float(smape(y_true.to_numpy(), prediction)),
        "WAPE": float(wape(y_true.to_numpy(), prediction)),
        "r2_score": float(r2_score(y_true, prediction)),
    }


def build_model_params(
    trial: optuna.Trial | None,
    feature_names: list[str],
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "monotone_constraints": [-1 if column == "price" else 0 for column in feature_names],
    }

    if trial is None:
        params.update(
            {
                "num_leaves": 31,
                "learning_rate": 0.03,
                "min_child_samples": 20,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
                "bagging_freq": 1,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
                "max_bin": 255,
            }
        )
        return params

    params.update(
        {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "max_bin": trial.suggest_int("max_bin", 63, 255),
        }
    )
    return params


def tune_hyperparameters(
    train: pd.DataFrame,
    target: str,
    n_trials: int,
    timeout: int | None,
    cv_splits: int,
    max_estimators: int,
    early_stopping_rounds: int,
) -> dict[str, Any]:
    feature_names = list(FEATURES)
    if n_trials <= 0:
        return build_model_params(None, feature_names)

    def objective(trial: optuna.Trial) -> float:
        params = build_model_params(trial, feature_names)
        scores: list[float] = []

        for fold_train, fold_validation in iter_date_splits(train, cv_splits):
            model = lgb.LGBMRegressor(**params, n_estimators=max_estimators)
            model.fit(
                fold_train[feature_names],
                fold_train[target],
                eval_set=[(fold_validation[feature_names], fold_validation[target])],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
            prediction = np.clip(model.predict(fold_validation[feature_names]), 0.0, None)
            scores.append(float(smape(fold_validation[target].to_numpy(), prediction)))

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return build_model_params(study.best_trial, feature_names)


def register_model(
    model: lgb.LGBMRegressor,
    model_name: str,
    model_alias: str,
) -> str:
    model_info = mlflow.lightgbm.log_model(
        model,
        artifact_path="model",
        registered_model_name=model_name,
        await_registration_for=120,
        pip_requirements=[
            "mlflow==2.13.2",
            "lightgbm==4.6.0",
            "numpy==1.26.4",
            "pandas==2.2.3",
            "scikit-learn==1.5.2",
        ],
    )
    version = getattr(model_info, "registered_model_version", None)
    if version is None:
        versions = MlflowClient().search_model_versions(f"name='{model_name}'")
        if not versions:
            raise RuntimeError(f"MLflow did not register model {model_name}")
        version = max(versions, key=lambda item: int(item.version)).version

    MlflowClient().set_registered_model_alias(
        name=model_name,
        alias=model_alias,
        version=str(version),
    )
    return str(version)


def train(
    train_path: str | Path = DEFAULT_DATA_PATH,
    target: str = "num_purchases",
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    n_trials: int = 20,
    timeout: int | None = 1800,
    cv_splits: int = 5,
    max_estimators: int = 1000,
    early_stopping_rounds: int = 50,
    model_name: str = DEFAULT_MODEL_NAME,
    model_alias: str = DEFAULT_MODEL_ALIAS,
    model_output: str | Path = DEFAULT_MODEL_OUTPUT,
) -> dict[str, Any]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "dynamic-pricing-demand")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    data = load_training_data(train_path, target)
    train_data, holdout_data, cutoff_date = temporal_holdout_split(data, train_fraction)
    feature_names = list(FEATURES)
    elasticity = calculate_price_elasticity(train_data, target)

    best_params = tune_hyperparameters(
        train=train_data,
        target=target,
        n_trials=n_trials,
        timeout=timeout,
        cv_splits=cv_splits,
        max_estimators=max_estimators,
        early_stopping_rounds=early_stopping_rounds,
    )

    with mlflow.start_run(run_name="lightgbm-demand-training") as run:
        mlflow.set_tags(
            {
                "task_type": "demand forecast for dynamic pricing",
                "framework": "lightgbm",
                "split_type": "temporal",
            }
        )
        mlflow.log_params(
            {
                "target": target,
                "train_path": str(Path(train_path)),
                "train_fraction": train_fraction,
                "cutoff_date": cutoff_date.date().isoformat(),
                "train_rows": len(train_data),
                "holdout_rows": len(holdout_data),
                "cv_splits": cv_splits,
                "optuna_trials": n_trials,
                "features": json.dumps(feature_names),
                "price_monotone_constraint": -1,
            }
        )
        mlflow.log_params({f"model_{key}": value for key, value in best_params.items()})
        if np.isfinite(elasticity):
            mlflow.log_metric("observed_price_elasticity", elasticity)

        evaluation_model = lgb.LGBMRegressor(**best_params, n_estimators=max_estimators)
        evaluation_model.fit(
            train_data[feature_names],
            train_data[target],
            eval_set=[(holdout_data[feature_names], holdout_data[target])],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        holdout_prediction = evaluation_model.predict(holdout_data[feature_names])
        metrics = regression_metrics(holdout_data[target], holdout_prediction)
        mlflow.log_metrics({f"holdout_{name}": value for name, value in metrics.items()})

        best_iteration = evaluation_model.best_iteration_ or max_estimators
        mlflow.log_param("best_iteration", best_iteration)

        final_model = lgb.LGBMRegressor(**best_params, n_estimators=best_iteration)
        final_model.fit(data[feature_names], data[target])

        output_path = Path(model_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_model.booster_.save_model(output_path)
        mlflow.log_artifact(str(output_path), artifact_path="native_model")
        mlflow.log_artifact(str(Path(train_path)), artifact_path="data")

        version = register_model(final_model, model_name, model_alias)

    result = {
        "run_id": run.info.run_id,
        "model_name": model_name,
        "model_version": version,
        "model_alias": model_alias,
        "cutoff_date": cutoff_date.date().isoformat(),
        "best_iteration": int(best_iteration),
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and register the dynamic-pricing demand model.")
    parser.add_argument("--train-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--target", default="num_purchases")
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--max-estimators", type=int, default=1000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--model-output", default=str(DEFAULT_MODEL_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        train_path=args.train_path,
        target=args.target,
        train_fraction=args.train_fraction,
        n_trials=args.n_trials,
        timeout=args.timeout,
        cv_splits=args.cv_splits,
        max_estimators=args.max_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        model_name=args.model_name,
        model_alias=args.model_alias,
        model_output=args.model_output,
    )


if __name__ == "__main__":
    main()
