# Dynamic Pricing System for E-Commerce

ML-сервис для прогноза спроса и подбора цены SKU с учетом GMV и маржи.

## Бизнес-идея

Проект решает задачу динамического ценообразования для онлайн-ретейла:

1. По SKU, дате, промо и товарным признакам прогнозируется спрос.
2. Для текущей цены строится набор цен-кандидатов в допустимом диапазоне.
3. Для каждой цены оцениваются ожидаемый спрос, GMV и маржа.
4. Выбирается цена с максимальным бизнес-score: GMV со штрафом за отклонение от целевой маржи.

Такой подход позволяет искать цену, которая увеличивает оборот, но не игнорирует маржинальность.

## Что внутри

- `application/` - исследовательская и training-часть: подготовка датасета, notebooks, обучение LightGBM/XGBoost, MLflow-логи.
- `sku_price_model_service/` - FastAPI-сервис для прогноза спроса и оптимизации цены.
- `frontend_ml/` - Streamlit-интерфейс для загрузки CSV и просмотра результатов.
- `docker-compose.yml` - локальная инфраструктура: PostgreSQL, MinIO, MLflow, FastAPI и Streamlit.

## API

FastAPI поднимается на `http://localhost:8005`.

- `GET /health` - базовая проверка, что API запущен.
- `POST /invocation` - принимает CSV и возвращает прогноз спроса в колонке `num_purchases`.
- `POST /optimize_price` - принимает CSV и возвращает оптимальную цену, ожидаемый спрос, GMV, маржу и базовые метрики.

Минимальные входные колонки CSV:

- `dates`
- `SKU`
- `price_per_sku`

Остальные признаки подтягиваются из таблиц PostgreSQL: `promo`, `sku_dict`, `prices`.

## Локальный запуск

1. Создать `.env` из примера:

```bash
cp .env.example .env
```

2. Поднять сервисы:

```bash
docker compose up -d --build
```

3. Загрузить demo-таблицы `promo`, `sku_dict`, `prices` в PostgreSQL:

```bash
docker compose run --rm seed
```

4. Обучить и зарегистрировать модель спроса:

```bash
docker compose run --rm trainer
```

Для быстрой проверки pipeline без полного подбора гиперпараметров:

```bash
docker compose run --rm trainer --n-trials 1 --cv-splits 3 --max-estimators 300
```

5. Открыть интерфейс:

```text
http://localhost:8501
```

Полезные локальные адреса:

- Streamlit UI: `http://localhost:8501`
- FastAPI: `http://localhost:8005`
- MLflow UI: `http://localhost:5001`

## ML-логика

Модель спроса обучается на `application/notebooks/data.csv`. Последние 10% дат используются как holdout, более ранние даты - для time-series cross-validation и Optuna. Колонка `margin` не попадает в признаки, а цена имеет монотонное ограничение `-1`: при прочих равных рост цены не должен повышать прогноз спроса.

После оценки на holdout модель переобучается на полном датасете, регистрируется в MLflow как `lgb_for_inference` и получает alias `champion`.

При оптимизации цена перебирается в диапазоне `70%..130%` от базовой цены. Для кандидата считается:

```text
GMV = candidate_price * expected_demand
margin = (candidate_price - cost) / candidate_price
score = GMV * (1 - lambda * max(0, target_margin - margin))
```

По умолчанию `target_margin = 0.5`, `lambda = 0.5`, количество кандидатов цены - `30`.

## Данные

- `application/notebooks/data.csv` - полный датасет из 6699 наблюдений; это источник для обучения.
- `application/train.csv` - историческая производная выборка из первых 90% строк. Канонический pipeline ее не использует.
- `application/data/raw/*.csv.dvc` и `application/data/processed/*.csv.dvc` - DVC-указатели, а не сами CSV.

## Важные замечания

- Demo-таблицы PostgreSQL загружаются командой `docker compose run --rm seed`.
- Модель обучается и регистрируется командой `docker compose run --rm trainer`.
- В репозитории исторически присутствуют MLflow/MinIO/IDE/venv-артефакты. Новые такие файлы игнорируются через `.gitignore`, но уже отслеживаемые артефакты лучше выносить отдельным cleanup-коммитом.
- Бизнес-смысл проекта не в ручном назначении цены, а в сравнении ценовых сценариев по спросу, GMV и марже.
