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
docker compose up --build
```

3. Загрузить demo-таблицы `promo`, `sku_dict`, `prices` в PostgreSQL:

```bash
docker compose run --rm seed
```

4. Открыть интерфейс:

```text
http://localhost:8501
```

Полезные локальные адреса:

- Streamlit UI: `http://localhost:8501`
- FastAPI: `http://localhost:8005`
- MLflow UI: `http://localhost:5001`

## ML-логика

Модель спроса использует признаки SKU, даты, промо, категории, бренда, поставщика и цены. При оптимизации цена перебирается в диапазоне `70%..130%` от базовой цены. Для кандидата считается:

```text
GMV = candidate_price * expected_demand
margin = (candidate_price - cost) / candidate_price
score = GMV * (1 - lambda * max(0, target_margin - margin))
```

По умолчанию `target_margin = 0.5`, `lambda = 0.5`, количество кандидатов цены - `30`.

## Важные замечания

- Для полноценного запуска нужны данные в PostgreSQL и зарегистрированная MLflow-модель `lgb_for_inference` в stage `Staging`.
- Demo-таблицы PostgreSQL можно загрузить командой `docker compose run --rm seed`; MLflow-модель эта команда не обучает и не регистрирует.
- В репозитории исторически присутствуют MLflow/MinIO/IDE/venv-артефакты. Новые такие файлы игнорируются через `.gitignore`, но уже отслеживаемые артефакты лучше выносить отдельным cleanup-коммитом.
- Бизнес-смысл проекта не в ручном назначении цены, а в сравнении ценовых сценариев по спросу, GMV и марже.
