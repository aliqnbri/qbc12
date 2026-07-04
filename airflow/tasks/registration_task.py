# File: app/airflow/tasks/registration_task.py
"""
Airflow task: Register best model to MLflow Model Registry.
"""
from datetime import datetime

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient


def task_register_model(
    experiment_name: str = "olist_late_delivery",
    model_name: str = "late_delivery_predictor",
    stage: str = "Staging",
    **context
) -> dict:
    """
    Airflow task: Register best model from experiment to Model Registry.
    
    Args:
        experiment_name: MLflow experiment name
        model_name: Name for registered model
        stage: Target stage (Staging/Production)
        context: Airflow context
    
    Returns:
        Registration metadata
    """
    task_start = datetime.now()
    logger.info(f"Starting model registration: {model_name}")
    
    client = MlflowClient()
    
    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")
    
    # Find best run by ROC-AUC
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError(f"No runs found in experiment: {experiment_name}")
    
    best_run = runs[0]
    run_id = best_run.info.run_id
    roc_auc = best_run.data.metrics.get("roc_auc", 0)
    
    logger.info(f"Best run: {run_id}, ROC-AUC={roc_auc:.4f}")
    
    # Register model
    model_uri = f"runs:/{run_id}/model"
    
    try:
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        version = registered_model.version
        
        # Transition to target stage
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
        logger.success(f"Registered {model_name} v{version} to stage {stage}")
    
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise
    
    task_end = datetime.now()
    execution_time = (task_end - task_start).total_seconds()
    
    metadata = {
        "task": "register_model",
        "model_name": model_name,
        "version": version,
        "stage": stage,
        "run_id": run_id,
        "roc_auc": roc_auc,
        "execution_time_seconds": execution_time,
        "completed_at": task_end.isoformat()
    }
    
    return metadata
