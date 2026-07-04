from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

# Import real tasks from your task modules
from app.airflow.tasks.ingestion_tasks import task_load_raw_data
from app.airflow.tasks.customer_feature_task import task_build_customer_features
from app.airflow.tasks.training_task import task_train_model
from app.airflow.tasks.registration_task import task_register_model
from app.airflow.tasks.smoke_test_task import task_api_smoke_test

with DAG(
    dag_id="group04_capstone_workflow",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["qbc12", "group04", "customer-level"],
) as dag:

    # 1. Load Data
    load = PythonOperator(
        task_id="load_raw_data", 
        python_callable=task_load_raw_data
    )

    # 2. Build Customer Features (The new requirement)
    features = PythonOperator(
        task_id="build_customer_features", 
        python_callable=task_build_customer_features,
        op_kwargs={
            "source_schema": "processed",
            "target_schema": "features",
            "target_table": "customer_features",
            "feature_version": "customer_v1"
        }
    )

    # 3. Train Model
    train = PythonOperator(
        task_id="train_model", 
        python_callable=task_train_model,
        op_kwargs={
            "feature_schema": "features",
            "feature_table": "customer_features",
            "experiment_name": "olist_customer_behavior"
        }
    )

    # 4. Register Model
    register = PythonOperator(
        task_id="register_model", 
        python_callable=task_register_model,
        op_kwargs={
            "experiment_name": "olist_customer_behavior",
            "model_name": "customer_behavior_predictor",
            "stage": "Staging"
        }
    )

    # 5. API Smoke Test
    smoke = PythonOperator(
        task_id="api_smoke_test", 
        python_callable=task_api_smoke_test
    )

    # Define dependencies
    load >> features >> train >> register >> smoke
