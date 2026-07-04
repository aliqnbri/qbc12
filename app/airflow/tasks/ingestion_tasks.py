# app/pipelines/tasks/ingestion_tasks.py
"""Airflow task wrappers for ingestion pipeline."""

from __future__ import annotations
import logging
from typing import Any
from app.pipelines.ingestion.manager import IngestionManager
from app.core.database import create_schemas_and_tables  
import app.models 

logger = logging.getLogger(__name__)


def task_load_raw_data(**context: Any) -> dict[str, Any]:
    """
    Airflow task: Load raw CSV files into PostgreSQL.
    
    Auto-discovers CSV files and matches them to table definitions
    using the normalize_dataset_name logic.
    
    """
    logger.info("Airflow Task: Starting raw data ingestion")
    create_schemas_and_tables()
    manager = IngestionManager(
        data_dir=None,  # Use settings.raw_data_dir
        truncate_before_load=True,
        fail_fast=True,
    )
    
    summary = manager.run(validate_only=False)
    
    if summary["failed_count"] > 0:
        raise RuntimeError(
            f"Raw ingestion failed for {summary['failed_count']} table(s): "
            f"{', '.join(summary['failed_tables'])}"
        )
    
    logger.info(
        "Airflow Task: Raw ingestion completed — "
        "%d tables, %d rows inserted",
        summary["success_count"],
        summary["total_rows_inserted"],
    )
    
    return {
        "status": summary["status"],
        "success_count": summary["success_count"],
        "failed_count": summary["failed_count"],
        "total_rows_inserted": summary["total_rows_inserted"],
        "failed_tables": summary["failed_tables"],
    }
