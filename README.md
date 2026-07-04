# Olist Late-Delivery Prediction — MLOps Capstone

End-to-end MLOps pipeline predicting whether an Olist order will be delivered
late (`order_delivered_customer_date > order_estimated_delivery_date`),
early enough for an operations team to intervene.

**Stack:** Python 3.12 · Polars · scikit-learn · XGBoost · MLflow · FastAPI ·
PostgreSQL · SQLAlchemy · Airflow · Prometheus · Grafana · Docker Compose ·
Pytest · Pydantic v2

## Architecture

```
raw CSVs ──> raw schema (PostgreSQL)
    │
    ├─ cleaning (dedupe, IQR capping, imputation) ──> processed schema
    │
    ├─ feature engineering (Polars, purchase-time only) ──> processed.order_features
    │
    ├─ training (RandomForest vs XGBoost, temporal split) ──> MLflow runs
    │
    ├─ registration (best by ROC AUC) ──> MLflow registry (Staging)
    │
    └─ FastAPI serving ──> Prometheus /metrics ──> Grafana dashboard
                          (Airflow orchestrates every step)
```

### Temporal-leakage policy

Only purchase-time information is used: order/payment/product/geography
attributes, the promised delivery window, and *historical* seller/customer
statistics computed strictly on the training window (chronological 80/20
split). Reviews and actual delivery timestamps are never features — they only
build the target. The API request schema rejects outcome fields outright
(`extra="forbid"`).

## Project structure

```
app/                  FastAPI service
  api/routers/        /, /health, /model-info, /predict, /predict-batch, /metrics
  api/schemas/        Pydantic v2 request/response models
  core/               config (env-driven), structured exceptions
  services/           model lifecycle + prediction orchestration
ml/                   cleaning, feature engineering, preprocessing,
                      training, evaluation, registry, inference policy
database/             SQLAlchemy 2.0 models, session, CRUD, loaders,
                      migrations/001_init.sql
utils/                dataset loader, logging, Prometheus metrics
airflow/dags/         olist_late_delivery_dag.py
docker/               mlflow/airflow images, postgres init, prometheus,
                      grafana provisioning + dashboard JSON
tests/                pytest suite (API, features, preprocessing, model)
```

## Installation (local)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit the credentials
```

Place the Olist CSVs (e.g. `olist_orders_dataset.csv`, ...) in `data/`.

## Environment variables

Everything is configured through `.env` — see `.env.example` for the full
annotated list. Key variables:

| Variable | Purpose |
|---|---|
| `POSTGRES_HOST/PORT/USER/PASSWORD/DB` | Application database |
| `AIRFLOW_DB`, `MLFLOW_DB` | Extra databases created at Postgres init |
| `MLFLOW_TRACKING_URI` | Tracking/registry server |
| `MODEL_NAME`, `MODEL_STAGE` | Registered model served by the API |
| `MODEL_LOCAL_PATH` | Fallback joblib artifact when the registry is down |
| `ENABLE_DB_LOGGING` | Persist served predictions to `processed.predictions` |
| `AIRFLOW_FERNET_KEY`, `AIRFLOW_WEBSERVER_SECRET_KEY` | Airflow secrets |
| `GRAFANA_ADMIN_USER/PASSWORD` | Grafana admin login |

No credential is hardcoded anywhere; `.env` is git-ignored.

## Docker (full platform)

```bash
cp .env.example .env    # fill in passwords + Airflow keys first
docker compose up -d --build
```

| Service | URL |
|---|---|
| API | http://localhost:8000 (docs at `/docs`) |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8080 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

## Database setup

Schemas/tables are created automatically by the loaders (ORM `create_all`),
and the equivalent DDL is versioned in `database/migrations/001_init.sql`:

```bash
psql "$DATABASE_URL" -f database/migrations/001_init.sql   # optional, idempotent
python -m database.loader --data-dir data --with-processed # raw + processed load
```

## Running Airflow

The `olist_late_delivery_pipeline` DAG (daily, no catchup) runs:

`load_raw_data → clean_raw_data → build_features → train_model →
register_model → api_smoke_test`

With Docker Compose, Airflow is initialized automatically (`airflow-init`).
Log in at http://localhost:8080 with `AIRFLOW_ADMIN_USER/PASSWORD`, unpause
the DAG and trigger it. Pipeline tasks execute in an isolated virtualenv
(`/opt/project-venv`) via `ExternalPythonOperator`, because the project's
SQLAlchemy 2.x stack conflicts with Airflow's own pins.

## Running MLflow

Compose starts an MLflow server backed by PostgreSQL with proxied artifacts.
For a quick local server instead:

```bash
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db --serve-artifacts
```

## Training

```bash
python -m ml.train --data-dir data --register
```

Trains a RandomForest and XGBoost (HistGradientBoosting fallback when xgboost
is unavailable) on a chronological split; logs parameters, the full metric
suite (ROC AUC, PR AUC, precision, recall, F1, confusion-matrix cells),
confusion-matrix JSON/PNG, feature importances and the fitted
preprocessing+model pipeline to MLflow; registers the ROC-AUC winner and
promotes it to `Staging`.

## Running the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

At startup the service loads `models:/$MODEL_NAME/$MODEL_STAGE`; if the
registry is unreachable it falls back to `MODEL_LOCAL_PATH`, then to a
deterministic heuristic baseline (the `olist_model_loaded` gauge drops to 0
so the degradation is observable).

### Prediction

```bash
curl -s http://localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "order_id": "abc-123",
  "order_purchase_timestamp": "2018-06-01T14:30:00",
  "order_estimated_delivery_date": "2018-06-20T00:00:00",
  "customer_state": "SP", "seller_state": "RJ",
  "price": 129.9, "freight_value": 19.9, "item_count": 1,
  "payment_type": "credit_card", "payment_installments": 3,
  "product_weight_g": 800, "product_length_cm": 30,
  "product_height_cm": 10, "product_width_cm": 20
}'
```

Response fields: `order_id`, `late_probability` (and its
`late_delivery_probability` alias), `risk_level` (`low` < 0.3 ≤ `medium`
< 0.6 ≤ `high`), `model_name`, `model_version`, `recommended_action`,
`latency_seconds`. Batch scoring: `POST /predict-batch` with
`{"orders": [...]}` (alias: `POST /batch-predict`).

Other endpoints: `GET /`, `GET /health`, `GET /model-info`,
`GET /metrics-summary`, `GET /metrics`, `GET /docs`.

## Tests

```bash
pytest -q tests
```

Covers the API contract (all endpoints, validation, leakage rejection,
determinism), feature engineering (target correctness, chronological split,
no forbidden features), the preprocessing pipeline (null imputation, unseen
categories) and model loading/fallback behaviour. Tests are hermetic — no
database, registry or network required.

## Prometheus

`GET /metrics` exposes: `olist_http_requests_total`,
`olist_request_latency_seconds`, `olist_predictions_total{risk_level}`,
`olist_prediction_errors_total{error_type}`,
`olist_prediction_latency_seconds`, `olist_predicted_probability` and
`olist_model_loaded`. Prometheus scrapes the API every 15 s
(`docker/prometheus/prometheus.yml`).

## Grafana

The provisioned dashboard **“Olist Late-Delivery API”**
(`docker/grafana/dashboards/olist_api_dashboard.json`, auto-loaded into the
*MLOps* folder) shows request rate per endpoint, HTTP error rate, latency
p50/p95/p99, inference latency, prediction throughput and risk distribution,
the predicted-probability heatmap and the model-loaded status.

## Datasets

The loader normalizes file names (`olist_orders_dataset.csv → orders`) via
`utils/normalize_dataset.py`. The pre-engineered `final` dataset is **never**
used for training — models are built from the raw tables; `final` may serve
only as a validation/comparison reference.
