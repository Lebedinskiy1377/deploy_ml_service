import os

from mlflow import MlflowClient

from metrics import wape
import fire
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import lightgbm as lgb
import optuna
from dotenv import load_dotenv


load_dotenv()
RANDOM_STATE = 42
TRAIN_SIZE = 0.9

remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri("http://localhost:5000")
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"


def job(
        train_path: str = "data/processed/data_train.csv",
        target: str = "num_purchases",
):
    """Model training job

    Args:
        train_path (str): Train dataset path
        target (str): Target column name
    """
    client = MlflowClient()

    # Начало эксперимента
    with mlflow.start_run():
        mlflow.set_tag("task_type", "sku prices forecast")
        mlflow.set_tag("framework", "lightgbm")

        df = pd.read_csv(train_path).rename(columns={'price_per_sku': 'price'})
        df = df.sort_values(['dates', 'SKU'])
        df['day'] = pd.to_datetime(df.dates).dt.day.astype('int64')
        df['month'] = pd.to_datetime(df.dates).dt.month.astype('int64')
        df['week_num'] = pd.to_datetime(df.dates).dt.isocalendar().week.astype('int64')
        df['weekday'] = pd.to_datetime(df.dates).dt.weekday.astype('int64')
        df.price *= 10
        train = df.head(int(len(df) * TRAIN_SIZE))
        test = df.tail(len(df) - int(len(df) * TRAIN_SIZE))

        X_train, y_train = train.drop(
            columns=['dates',
                     'num_purchases',
                     'creation_date',
                     'expiration_date'
                     ]), train.num_purchases
        X_test, y_test = test.drop(
            columns=['dates', 'num_purchases', 'creation_date', 'expiration_date']), test.num_purchases

        for col, coltype in zip(X_train.columns, X_train.dtypes):
            if coltype == 'object':
                X_train[col] = X_train[col].astype('category')
                X_test[col] = X_test[col].astype('category')

        mlflow.log_param('features', list(X_train.columns))
        mlflow.log_param('target', target)

        print(X_train.columns)
        print(X_train.dtypes)

        tscv = TimeSeriesSplit(n_splits=5)

        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),
                'num_leaves': trial.suggest_int('num_leaves', 20, 400),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'max_bin': trial.suggest_int('max_bin', 32, 1023),
                'feature_contri': [5.0 if col == 'price' else 1.0 for col in X_train.columns],
                'monotone_constraints': [1 if col == 'price' else 0 for col in X_train.columns],
                'verbosity': -1,
            }

            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train, y_train):
                X_train_, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = lgb.LGBMRegressor(**params, n_estimators=1000)
                model.fit(
                    X_train_, y_train_,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )

                y_pred = model.predict(X_val)
                cv_scores.append(wape(y_val, y_pred))

            return np.mean(cv_scores)

        study_lgb = optuna.create_study(direction='minimize')
        study_lgb.optimize(objective, n_trials=100, timeout=3600)

        study_lgb.best_params['feature_contri'] = [5.0 if col == 'price' else 1.0 for col in X_train.columns]
        study_lgb.best_params['monotone_constraints'] = [1 if col == 'price' else 0 for col in X_train.columns]
        study_lgb.best_params['verbosity'] = 0

        model = lgb.LGBMRegressor(**study_lgb.best_params, n_estimators=1000)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        booster = model.booster_
        booster.save_model("lgb_model.txt")

        y_pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("MAPE", mape)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("r2_score", r2)

        mlflow.log_artifact(train_path, "data")

        try:
            mlflow.lightgbm.log_model(
                model,
                artifact_path="model",
                registered_model_name="lgb_for_inference",
            )
            print(f"Model registered as 'lgb_for_inference'")

            latest_version = client.get_latest_versions("lgb_for_inference", stages=["None"])[0].version
            client.transition_model_version_stage(
                name="lgb_for_inference",
                version=latest_version,
                stage="Staging"
            )
            print(f"Model version {latest_version} transitioned to Staging")

        except Exception as e:
            print(f"Failed to register model: {e}")


if __name__ == "__main__":
    fire.Fire(job)
