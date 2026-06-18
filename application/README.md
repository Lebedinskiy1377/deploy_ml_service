# Training Application

Исследовательская и training-часть проекта динамического ценообразования.

## Назначение

Модуль готовит данные по SKU, обучает модель спроса и регистрирует результат в MLflow для дальнейшего использования FastAPI-сервисом.

Основная целевая переменная:

- `num_purchases` - спрос в штуках.

Основные признаки:

- дата и календарные признаки;
- SKU;
- цена;
- промо/скидка;
- товарная иерархия;
- поставщик и бренд.

## Структура

- `src/data/` - сборка и очистка датасетов.
- `src/models/` - обучение, метрики и offline prediction.
- `data/raw/` и `data/processed/` - DVC-ссылки на данные.
- `notebooks/` - исследовательские эксперименты.
- `mlruns/` - исторические MLflow-артефакты.

## Обучение

Основной источник данных - `notebooks/data.csv`. Файл `train.csv` является старой производной выборкой из первых 90% строк и каноническим pipeline не используется.

Рекомендуемый запуск из корня репозитория:

```bash
docker compose run --rm trainer
```

Быстрый smoke-run:

```bash
docker compose run --rm trainer --n-trials 1 --cv-splits 3 --max-estimators 300
```

Локальный запуск после установки `requirements-train.txt`:

```bash
python -m src.models.train_model
```

Pipeline:

1. Проверяет схему, пропуски, дубли и допустимость цены/таргета.
2. Исключает `margin` и использует явный feature contract, совпадающий с FastAPI.
3. Делит данные по уникальным датам, не смешивая одну дату между train и validation.
4. Подбирает LightGBM через Optuna по SMAPE.
5. Использует монотонное ограничение `-1` для цены.
6. Считает MAE, RMSE, MAPE, SMAPE, WAPE и R2 на holdout.
7. Переобучает модель на полном датасете и регистрирует ее в MLflow.

Скрипт использует `MLFLOW_TRACKING_URI`, если переменная задана. Локальный дефолт:

```text
http://localhost:5000
```

Обученная модель регистрируется в MLflow как `lgb_for_inference` и получает alias `champion`.

Старый вход остается совместимым и запускает тот же pipeline:

```bash
python -m src.models.train_price
```
