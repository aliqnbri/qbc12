"""End-to-end Olist late-delivery pipeline DAG.

load_raw_data -> clean_raw_data -> build_features -> train_model
    -> register_model -> api_smoke_test

Pipeline tasks run through ``ExternalPythonOperator`` inside the dedicated
project virtualenv (``/opt/project-venv``) because the project's dependency
set (SQLAlchemy 2.x, MLflow, Polars) conflicts with Airflow's own pins.
Each callable is fully self-contained: it re-imports its dependencies and
reads its configuration from the environment, as required by the operator's
source-serialization model. ``PYTHONPATH`` must include the mounted project
root (``/opt/airflow/project``); see docker-compose.yml.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import ExternalPythonOperator, PythonOperator

PROJECT_PYTHON = os.environ.get("PROJECT_PYTHON", "/opt/project-venv/bin/python")

DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "depends_on_past": False,
}


def task_load_raw_data() -> dict:
    """Load every raw CSV into the ``raw`` PostgreSQL schema."""
    import os

    from database.loader import load_raw_to_db

    return load_raw_to_db(os.environ.get("DATA_DIR", "/opt/airflow/project/data"))


def task_clean_raw_data() -> dict:
    """Clean the raw datasets (dedupe, IQR capping, imputation, text
    normalization) and persist them into the ``processed`` schema."""
    import os

    from database.loader import write_processed_tables
    from ml.cleaning import clean_datasets
    from utils.normalize_dataset import load_datasets

    datasets = load_datasets(os.environ.get("DATA_DIR", "/opt/airflow/project/data"))
    cleaned = clean_datasets(datasets)
    return write_processed_tables(cleaned)


def task_build_features() -> int:
    """Engineer leakage-safe order features into ``processed.order_features``."""
    import os

    import polars as pl

    from database.loader import save_features_to_db
    from ml.cleaning import clean_datasets
    from ml.feature_engineering import build_full_feature_frame
    from utils.normalize_dataset import load_datasets

    datasets = clean_datasets(
        load_datasets(os.environ.get("DATA_DIR", "/opt/airflow/project/data"))
    )
    train_frame, test_frame = build_full_feature_frame(datasets)
    features = pl.concat([train_frame, test_frame], how="vertical")
    return save_features_to_db(features)


def task_train_model() -> str:
    """Train and compare candidates in MLflow; return the winning run id."""
    import os

    from ml.train import run_training

    result = run_training(
        data_dir=os.environ.get("DATA_DIR", "/opt/airflow/project/data")
    )
    return result.best.run_id


def task_register_model(best_run_id: str) -> dict:
    """Register the winning run in the MLflow registry and promote it."""
    from ml.registry import register_best_model

    if not best_run_id:
        raise ValueError("Empty best_run_id received from train_model")
    info = register_best_model(best_run_id)
    return {"name": info.name, "version": info.version, "stage": info.stage}


def task_api_smoke_test() -> None:
    """Verify the live API serves predictions with the required contract."""
    import os

    import requests

    api_url = os.environ.get("API_URL", "http://api:8000")

    health = requests.get(f"{api_url}/health", timeout=10)
    health.raise_for_status()

    response = requests.post(
        f"{api_url}/predict",
        timeout=15,
        json={
            "order_id": "smoke-test-0001",
            "order_purchase_timestamp": "2018-06-01T14:30:00",
            "order_estimated_delivery_date": "2018-06-20T00:00:00",
            "customer_state": "SP",
            "seller_state": "RJ",
            "price": 129.9,
            "freight_value": 19.9,
            "item_count": 1,
            "payment_type": "credit_card",
            "payment_installments": 3,
            "product_weight_g": 800,
            "product_length_cm": 30,
            "product_height_cm": 10,
            "product_width_cm": 20,
        },
    )
    response.raise_for_status()
    body = response.json()
    required_keys = {
        "order_id",
        "late_probability",
        "risk_level",
        "model_name",
        "model_version",
        "recommended_action",
    }
    missing = required_keys - set(body)
    if missing:
        raise AssertionError(f"Prediction response missing keys: {sorted(missing)}")
    if not 0.0 <= body["late_probability"] <= 1.0:
        raise AssertionError(f"Probability out of range: {body['late_probability']}")


with DAG(
    dag_id="olist_late_delivery_pipeline",
    description="Load, clean, feature-engineer, train, register and smoke-test",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["olist", "mlops", "late-delivery"],
) as dag:
    load_raw_data = ExternalPythonOperator(
        task_id="load_raw_data",
        python=PROJECT_PYTHON,
        python_callable=task_load_raw_data,
        expect_airflow=False,
    )
    clean_raw_data = ExternalPythonOperator(
        task_id="clean_raw_data",
        python=PROJECT_PYTHON,
        python_callable=task_clean_raw_data,
        expect_airflow=False,
    )
    build_features = ExternalPythonOperator(
        task_id="build_features",
        python=PROJECT_PYTHON,
        python_callable=task_build_features,
        expect_airflow=False,
    )
    train_model = ExternalPythonOperator(
        task_id="train_model",
        python=PROJECT_PYTHON,
        python_callable=task_train_model,
        expect_airflow=False,
    )
    register_model = ExternalPythonOperator(
        task_id="register_model",
        python=PROJECT_PYTHON,
        python_callable=task_register_model,
        op_args=["{{ ti.xcom_pull(task_ids='train_model') }}"],
        expect_airflow=False,
    )
    api_smoke_test = PythonOperator(
        task_id="api_smoke_test",
        python_callable=task_api_smoke_test,
    )

    (
        load_raw_data
        >> clean_raw_data
        >> build_features
        >> train_model
        >> register_model
        >> api_smoke_test
    )
