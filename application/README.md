# Training

Обучение модели спроса для динамического ценообразования. Общая картина, запуск всего стека и API описаны в [README в корне](../README.md).

Целевая переменная — `num_purchases` (спрос в штуках). Признаки: календарь, SKU, цена, промо-скидка, товарная иерархия, поставщик, бренд, даты заведения и вывода SKU. Список признаков (`FEATURES` в `src/models/train_model.py`) совпадает с `sku_price_model_service/app/config.py`, это проверяют тесты.

## Запуск

Через Docker, из корня репозитория:

```bash
docker compose run --rm --build trainer                    # полное обучение
docker compose run --rm --build trainer --n-trials 1 --cv-splits 3 --max-estimators 300
```

На хосте, из этой папки (`pip install -r requirements.txt`):

```bash
python -m src.models.train_model --help
```

MLflow берётся из `MLFLOW_TRACKING_URI`, по умолчанию `http://localhost:5001` — MLflow из `docker compose`. Скрипт читает `.env` из корня репозитория, если он есть.

## Pipeline

1. Проверяет схему, пропуски, дубли `dates`/`SKU`, положительность цены и таргета.
2. Делит данные по уникальным датам: последние 10% — holdout, одна дата не попадает в обе части.
3. Подбирает гиперпараметры LightGBM через Optuna по SMAPE на `TimeSeriesSplit`.
4. Ставит монотонное ограничение `-1` на цену; `margin` в признаки не входит.
5. Считает MAE, RMSE, MAPE, SMAPE, WAPE и R2 на holdout.
6. Переобучает модель на всём датасете, регистрирует `lgb_for_inference` и ставит alias `champion`.

## Данные

- `data/processed/sku_sales.csv` — датасет для обучения (6699 строк, 25 SKU).
- `data/raw/*.dvc`, `data/processed/*.dvc` — DVC-указатели на сырые выгрузки; как получить доступ, описано в корневом README.
- `src/data/make_dataset.py` — сборка объединённого датасета из сырых выгрузок; синтетическую колонку `margin` добавляет ноутбук `notebooks/sku.ipynb`.
- `notebooks/` — исследования: EDA, XGBoost и CatBoost baseline'ы. Зависимости: `requirements-notebooks.txt`.
