# File: app/airflow/tasks/feature_task.py
"""
Airflow task wrapper for feature engineering pipeline.
"""
from datetime import datetime
from typing import Optional

import polars as pl
from loguru import logger

from app.core.database import get_engine
from app.pipelines.feature_engineering.builder import FeatureBuilder


def task_build_features(
    source_schema: str = "processed",
    target_schema: str = "features",
    target_table: str = "order_features",
    feature_version: str = "v1",
    **context
) -> dict:
    """
    Airflow task: Build order-level feature table from processed tables.
    
    Args:
        source_schema: Schema containing processed tables
        target_schema: Schema for feature table output
        target_table: Name of feature table
        feature_version: Version tag for features
        context: Airflow context
    
    Returns:
        Task execution metadata
    """
    task_start = datetime.now()
    logger.info(f"Starting feature engineering task: {feature_version}")
    
    engine = get_engine()
    
    # Load processed tables as LazyFrames
    logger.info(f"Loading tables from {source_schema}")
    tables = {}
    table_names = [
        "orders", "order_items", "customers", "products",
        "sellers", "order_payments", "order_reviews"
    ]
    
    for table_name in table_names:
        query = f'SELECT * FROM "{source_schema}"."{table_name}"'
        tables[table_name] = pl.read_database_uri(
            query=query,
            uri=str(engine.url),
            engine="connectorx"
        ).lazy()
        logger.info(f"Loaded {table_name}: {tables[table_name].collect().shape}")
    
    # Build features
    builder = FeatureBuilder(feature_version=feature_version)
    features_lf = builder.engineer_features(tables)
    
    # Collect and write to database
    features_df = features_lf.collect()
    logger.info(f"Engineered features shape: {features_df.shape}")
    
    # Write to PostgreSQL
    full_table_name = f'"{target_schema}"."{target_table}"'
    features_df.write_database(
        table_name=full_table_name,
        connection=str(engine.url),
        if_table_exists="replace",
        engine="sqlalchemy"
    )
    logger.info(f"Written feature table to {full_table_name}")
    
    task_end = datetime.now()
    execution_time = (task_end - task_start).total_seconds()
    
    metadata = {
        "task": "build_features",
        "source_schema": source_schema,
        "target_schema": target_schema,
        "target_table": target_table,
        "feature_version": feature_version,
        "row_count": len(features_df),
        "column_count": len(features_df.columns),
        "execution_time_seconds": execution_time,
        "completed_at": task_end.isoformat()
    }
    
    logger.success(f"Feature engineering complete: {metadata}")
    return metadata
