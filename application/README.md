# Training

Trains the demand model for dynamic pricing. See the [root README](../README.md) for the whole stack and the API.

The target is `num_purchases` (units sold). Features: calendar, SKU, price, promo discount, product hierarchy, vendor, brand, SKU creation and expiration dates. The feature list (`FEATURES` in `src/models/train_model.py`) matches `sku_price_model_service/app/config.py`; tests keep them in sync.

## Run

With Docker, from the repository root:

```bash
docker compose run --rm --build trainer                    # full training
docker compose run --rm --build trainer --n-trials 1 --cv-splits 3 --max-estimators 300
```

On the host, from this folder (`pip install -r requirements.txt`):

```bash
python -m src.models.train_model --help
```

The MLflow server is taken from `MLFLOW_TRACKING_URI`, `http://localhost:5001` by default (MLflow from `docker compose`). The script reads `.env` from the repository root if it exists.

## Pipeline

1. Validates the schema, missing values, duplicate `dates`/`SKU` rows and that price and target are positive.
2. Splits by unique dates: the last 10% of dates form the holdout, so no date lands in both parts.
3. Tunes LightGBM hyperparameters with Optuna on `TimeSeriesSplit`, optimising SMAPE.
4. Applies a monotonic constraint (`-1`) on price; `margin` is not a feature.
5. Picks the number of trees by early stopping on the last 10% of the training period, refits on the whole training period and reports MAE, RMSE, MAPE, SMAPE, WAPE and R2 on the holdout. The holdout is used neither for tuning nor for early stopping.
6. Retrains on the full dataset, registers `lgb_for_inference` and sets the `champion` alias.

## Data

- `data/processed/sku_sales.csv`: the training dataset (6,699 rows, 25 SKUs).
- `data/raw/*.dvc`, `data/processed/*.dvc`: DVC pointers to the raw exports.
- `src/data/make_dataset.py`: builds the merged dataset from the raw exports; the notebook `notebooks/sku.ipynb` adds the synthetic `margin` column.
- `notebooks/`: research (EDA, XGBoost and CatBoost baselines). Dependencies: `requirements-notebooks.txt`.
