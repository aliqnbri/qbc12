# File: app/airflow/tasks/smoke_test_task.py
"""
Airflow task: Smoke test FastAPI prediction endpoint.
"""
from datetime import datetime
from typing import List

import httpx
import polars as pl
from loguru import logger

from app.core.database import get_engine


def task_api_smoke_test(
    api_base_url: str = "http://api:8000",
    num_test_samples: int = 5,
    **context
) -> dict:
    """
    Airflow task: Test FastAPI /predict endpoint with sample data.
    
    Args:
        api_base_url: Base URL of FastAPI service
        num_test_samples: Number of test requests to send
        context: Airflow context
    
    Returns:
        Test results metadata
    """
    task_start = datetime.now()
    logger.info(f"Starting API smoke test: {api_base_url}")
    
    # Load sample features from database
    engine = get_engine()
    query = f'SELECT * FROM "features"."order_features" LIMIT {num_test_samples}'
    
    samples_df = pl.read_database_uri(
        query=query,
        uri=str(engine.url),
        engine="connectorx"
    )
    
    logger.info(f"Loaded {len(samples_df)} test samples")
    
    # Test health endpoint
    try:
        with httpx.Client(timeout=10.0) as client:
            health_response = client.get(f"{api_base_url}/health")
            health_response.raise_for_status()
            logger.success(f"Health check passed: {health_response.json()}")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise
    
    # Test prediction endpoint
    results = []
    
    with httpx.Client(timeout=30.0) as client:
        for row in samples_df.iter_rows(named=True):
            try:
                response = client.post(
                    f"{api_base_url}/predict",
                    json=row
                )
                response.raise_for_status()
                prediction = response.json()
                
                results.append({
                    "order_id": row["order_id"],
                    "status": "success",
                    "late_probability": prediction["late_probability"],
                    "risk_level": prediction["risk_level"]
                })
                
                logger.info(f"Prediction success: order={row['order_id']}, prob={prediction['late_probability']:.3f}")
            
            except Exception as e:
                results.append({
                    "order_id": row.get("order_id", "unknown"),
                    "status": "error",
                    "error": str(e)
                })
                logger.error(f"Prediction failed: {e}")
    
    task_end = datetime.now()
    execution_time = (task_end - task_start).total_seconds()
    
    successful = len([r for r in results if r["status"] == "success"])
    
    metadata = {
        "task": "api_smoke_test",
        "api_base_url": api_base_url,
        "total_tests": len(results),
        "successful": successful,
        "failed": len(results) - successful,
        "success_rate": successful / len(results) if results else 0,
        "results": results,
        "execution_time_seconds": execution_time,
        "completed_at": task_end.isoformat()
    }
    
    if successful == 0:
        raise RuntimeError("All API smoke tests failed")
    
    logger.success(f"Smoke test complete: {successful}/{len(results)} passed")
    return metadata
