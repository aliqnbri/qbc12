"""
Olist MLOps Pipeline - Complete DAG
Group 04 - QBC12
"""
from datetime import datetime
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import task functions
from app.airflow.tasks.ingestion_tasks import task_load_raw_data
from app.airflow.tasks.cleaning_tasks import task_clean_raw_data
from app.airflow.tasks.feature_task import task_build_features
from app.airflow.tasks.export_dataset import run_export_dataset_task

# from app.airflow.tasks.training_task import task_train_model
# from app.airflow.tasks.registration_task import task_register_model
# from app.airflow.tasks.smoke_test_task import task_api_smoke_test


with DAG(
    dag_id="group04_capstone_workflow",
    description="End-to-end MLOps pipeline: ingestion → cleaning → features → training → registration → smoke test",
    start_date=datetime(2025, 1, 1),
    schedule=None,  # Manual trigger only
    catchup=False,
    tags=["qbc12", "group04", "mlops"],
    default_args={
        "owner": "group04",
        "retries": 0,
    },
) as dag:

    # ========== Task 1: Load Raw Data ==========
    load = PythonOperator(
        task_id="load_raw_data",
        python_callable=task_load_raw_data,
        doc_md="""
        Load all CSV files from data/raw/ into PostgreSQL raw schema.
        - Uses IngestionManager
        - Validates row counts
        - Returns summary with success/failed counts
        """,
    )

    # ========== Task 2: Clean Raw Data ==========
    clean = PythonOperator(
        task_id="clean_raw_data",
        python_callable=task_clean_raw_data,
        doc_md="""
        Clean all tables from raw → processed schema.
        - Handles nulls, outliers, temporal inconsistencies
        - Uses Polars + _pg_copy for fast writes
        - Returns cleaning summary
        """,
    )

    # # ========== Task 3: Build Features ==========
    features = PythonOperator(
        task_id="build_features",
        python_callable=task_build_features,
        op_kwargs={
            "source_schema": "processed",
            "target_schema": "features",
            "target_table": "order_features",
            "feature_version": "v1",
            "truncate": True
        },
        doc_md="Build order-level feature table with delay target"
        
    )

    # # ========== Task 4: export dvc ==========
    export = PythonOperator(
        task_id="export_dataset",
        python_callable=run_export_dataset_task,
        op_kwargs={"use_dvc": True, "compression": "snappy"},
        
        
)
    # # ========== Task 4: Train Model ==========
    # train = PythonOperator(
    #     task_id="train_model",
    #     python_callable=task_train_model,
    #     op_kwargs={
    #         "feature_schema": "features",
    #         "feature_table": "order_features",
    #         "experiment_name": "olist_late_delivery",
    #         "target_column": "is_late_delivery",
    #     },
    #     doc_md="""
    #     Train multiple models and log to MLflow.
    #     - Loads feature table from PostgreSQL
    #     - Trains at least 2 models (e.g., XGBoost, LightGBM)
    #     - Logs params, metrics, artifacts to MLflow
    #     - Returns best run_id
    #     """,
    # )

    # # ========== Task 5: Register Model ==========
    # register = PythonOperator(
    #     task_id="register_model",
    #     python_callable=task_register_model,
    #     op_kwargs={
    #         "experiment_name": "olist_late_delivery",
    #         "model_name": "late_delivery_predictor",
    #         "stage": "Staging",
    #     },
    #     doc_md="""
    #     Register best model to MLflow Model Registry.
    #     - Finds best run from experiment
    #     - Registers to Model Registry
    #     - Transitions to Staging
    #     - Returns model version
    #     """,
    # )

    # # ========== Task 6: API Smoke Test ==========
    # smoke = PythonOperator(
    #     task_id="api_smoke_test",
    #     python_callable=task_api_smoke_test,
    #     op_kwargs={
    #         "api_base_url": "http://fastapi:8000",
    #     },
    #     doc_md="""
    #     Test FastAPI endpoints after model registration.
    #     - GET /health
    #     - GET /model-info
    #     - POST /predict (with sample payload)
    #     - Validates response structure
    #     """,
    # )

    # ========== Task Dependencies ==========
    load >> clean >> features >> export
    # load >> clean >> features >> train >> register >> smoke
