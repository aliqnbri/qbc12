# File: app/airflow/tasks/feature_task.py
"""
Airflow task wrapper for order-level feature engineering pipeline.
"""
from datetime import datetime
from typing import Dict

import polars as pl
from loguru import logger

from app.core.database import get_sync_conn, _pg_copy
from app.pipelines.feature_engineering.builder import FeatureBuilder


def task_build_features(
    source_schema: str = "processed",
    target_schema: str = "features",
    target_table: str = "order_features",
    feature_version: str = "v1",
    truncate: bool = True,
    **context
) -> dict:
    """
    Airflow task: Build order-level feature table from processed tables.
    
    Pipeline:
        1. Load processed tables from PostgreSQL
        2. Pass to FeatureBuilder (pure transformation)
        3. Collect result
        4. Write to target schema using _pg_copy
    
    Args:
        source_schema: Schema containing processed tables (default: "processed")
        target_schema: Schema for feature table output (default: "features")
        target_table: Name of feature table (default: "order_features")
        feature_version: Version tag for features (default: "v1")
        truncate: Whether to truncate target table before insert
        context: Airflow context dict
    
    Returns:
        Task execution metadata with row/column counts and timing
    """
    task_start = datetime.now()
    logger.info(f"🚀 Starting order-level feature engineering task (version: {feature_version})")
    
    # Step 1: Load processed tables
    logger.info(f"📥 Loading tables from schema: {source_schema}")
    tables = _load_processed_tables(source_schema)
    
    # Step 2: Build features (pure transformation)
    logger.info(f"🔧 Running FeatureBuilder.engineer_features()")
    builder = FeatureBuilder(feature_version=feature_version)
    features_lf = builder.engineer_features(tables)
    
    # Step 3: Collect LazyFrame to DataFrame
    logger.info("⚡ Collecting LazyFrame...")
    features_df = features_lf.collect()
    logger.info(f"✅ Engineered features shape: {features_df.shape}")
    
    # Step 4: Write to database
    logger.info(f"💾 Writing to {target_schema}.{target_table}")
    rows_written = _write_features_to_db(
        df=features_df,
        schema=target_schema,
        table=target_table,
        truncate=truncate
    )
    
    # Metadata
    task_end = datetime.now()
    execution_time = (task_end - task_start).total_seconds()
    
    metadata = {
        "task": "build_features",
        "source_schema": source_schema,
        "target_schema": target_schema,
        "target_table": target_table,
        "feature_version": feature_version,
        "row_count": rows_written,
        "column_count": len(features_df.columns),
        "execution_time_seconds": round(execution_time, 2),
        "completed_at": task_end.isoformat()
    }
    
    logger.success(f"🎉 Feature engineering complete: {metadata}")
    return metadata


def _load_processed_tables(schema: str) -> Dict[str, pl.LazyFrame]:
    """
    Load all required processed tables from PostgreSQL as LazyFrames.
    
    Args:
        schema: Source schema name
    
    Returns:
        Dictionary mapping table names to LazyFrames
    """
    table_names = [
        "orders",
        "order_items",
        "customers",
        "products",
        "sellers",
        "order_payments",
        "order_reviews"
    ]
    
    tables = {}
    
    with get_sync_conn() as conn:
        for table_name in table_names:
            query = f'SELECT * FROM "{schema}"."{table_name}"'
            df = pl.read_database(query, conn)
            tables[table_name] = df.lazy()
            logger.debug(f"  ✓ Loaded {schema}.{table_name}: {len(df):,} rows")
    
    logger.info(f"📦 Loaded {len(tables)} tables from {schema}")
    return tables


def _write_features_to_db(
    df: pl.DataFrame,
    schema: str,
    table: str,
    truncate: bool
) -> int:
    """
    Write feature DataFrame to PostgreSQL using _pg_copy.
    
    Args:
        df: Feature DataFrame
        schema: Target schema
        table: Target table name
        truncate: Whether to truncate before insert
    
    Returns:
        Number of rows written
    """
    with get_sync_conn() as conn:
        rows_written = _pg_copy(
            df=df,
            schema=schema,
            table=table,
            conn=conn,
            truncate=truncate
        )
        logger.success(f"✅ Written {rows_written:,} rows to {schema}.{table}")
    
    return rows_written
