"""
Airflow task for customer-level feature engineering.
"""

from datetime import datetime
from loguru import logger
from app.pipelines.feature_engineering.customer_builder import CustomerFeatureBuilder


def task_build_customer_features(
    source_schema: str = "processed",
    target_schema: str = "features",
    target_table: str = "customer_features",
    feature_version: str = "customer_v1",
    **context
) -> dict:
    """
    Build customer-level aggregated features from processed tables.
    
    This task:
    - Reads from processed schema (orders, customers, etc.)
    - Joins and aggregates to customer_unique_id level
    - Writes to features.customer_features
    - Returns execution metadata for XCom
    
    Args:
        source_schema: Schema to read processed tables from
        target_schema: Schema to write features to
        target_table: Target table name
        feature_version: Feature version tag
        **context: Airflow context
        
    Returns:
        dict: Task execution metadata
    """
    task_start = datetime.now()
    
    logger.info(f"Starting customer feature engineering task")
    logger.info(f"Source: {source_schema}.*")
    logger.info(f"Target: {target_schema}.{target_table}")
    logger.info(f"Feature version: {feature_version}")
    
    try:
        # Initialize builder
        builder = CustomerFeatureBuilder(feature_version=feature_version)
        
        # Build features
        features_df = builder.build_features(
            source_schema=source_schema,
            target_schema=target_schema,
            target_table=target_table,
            truncate=True
        )
        
        task_end = datetime.now()
        execution_time = (task_end - task_start).total_seconds()
        
        result = {
            "task": "build_customer_features",
            "source_schema": source_schema,
            "target_table": f"{target_schema}.{target_table}",
            "feature_version": feature_version,
            "num_customers": len(features_df),
            "num_features": len(features_df.columns),
            "execution_time_seconds": round(execution_time, 2),
            "completed_at": task_end.isoformat()
        }
        
        logger.success(f"✅ Task completed successfully in {execution_time:.2f}s")
        logger.info(f"Generated features for {len(features_df):,} customers")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Task failed: {str(e)}")
        raise RuntimeError(f"Customer feature engineering failed: {str(e)}") from e
