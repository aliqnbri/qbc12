# # File: app/airflow/tasks/training_task.py
# """
# Airflow task wrapper for model training with MLflow.
# Uses existing train.py logic.
# """
# from datetime import datetime
# from typing import List

# import mlflow
# import polars as pl
# from loguru import logger

# from app.core.database import sync_engine
# import sys
# from train import train_model, MODELS_CONFIG  # Your existing training functions


# def task_train_model(
#     feature_schema: str = "features",
#     feature_table: str = "order_features",
#     experiment_name: str = "olist_late_delivery",
#     models: List[str] = None,
#     **context
# ) -> dict:
#     """
#     Airflow task: Train models and log to MLflow.
    
#     Args:
#         feature_schema: Schema containing feature table
#         feature_table: Name of feature table
#         experiment_name: MLflow experiment name
#         models: List of model names to train (from MODELS_CONFIG)
#         context: Airflow context
    
#     Returns:
#         Task execution metadata with best model info
#     """
#     task_start = datetime.now()
#     logger.info(f"Starting model training: experiment={experiment_name}")
    
#     if models is None:
#         models = ["logistic_regression", "random_forest"]
    
#     # Set MLflow experiment
#     mlflow.set_experiment(experiment_name)
    
#     # Load feature table
#     engine = sync_engine()
#     query = f'SELECT * FROM "{feature_schema}"."{feature_table}"'
#     features_df = pl.read_database_uri(
#         query=query,
#         uri=str(engine.url),
#         engine="connectorx"
#     )
    
#     logger.info(f"Loaded features: {features_df.shape}")
    
#     # Train each model
#     results = {}
#     best_model = None
#     best_score = -1
    
#     for model_name in models:
#         if model_name not in MODELS_CONFIG:
#             logger.warning(f"Unknown model: {model_name}, skipping")
#             continue
        
#         logger.info(f"Training model: {model_name}")
        
#         try:
#             # Call your existing train_model function
#             run_info = train_model(
#                 df=features_df,
#                 model_name=model_name,
#                 experiment_name=experiment_name
#             )
            
#             results[model_name] = run_info
            
#             # Track best model by ROC-AUC
#             if run_info["metrics"]["roc_auc"] > best_score:
#                 best_score = run_info["metrics"]["roc_auc"]
#                 best_model = model_name
            
#             logger.success(f"Trained {model_name}: ROC-AUC={run_info['metrics']['roc_auc']:.4f}")
        
#         except Exception as e:
#             logger.error(f"Failed to train {model_name}: {e}")
#             results[model_name] = {"error": str(e)}
    
#     task_end = datetime.now()
#     execution_time = (task_end - task_start).total_seconds()
    
#     metadata = {
#         "task": "train_model",
#         "experiment_name": experiment_name,
#         "models_trained": len([m for m in results if "error" not in results[m]]),
#         "best_model": best_model,
#         "best_roc_auc": best_score,
#         "results": results,
#         "execution_time_seconds": execution_time,
#         "completed_at": task_end.isoformat()
#     }
    
#     logger.success(f"Training complete: best={best_model}, score={best_score:.4f}")
#     return metadata
