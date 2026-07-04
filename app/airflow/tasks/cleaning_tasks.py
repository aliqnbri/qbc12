"""
Airflow task wrapper for data cleaning.
"""
from __future__ import annotations

import logging
from typing import Any

from app.pipelines.cleaning.manager import CleaningManager

logger = logging.getLogger(__name__)


def task_clean_raw_data(**context: Any) -> dict[str, Any]:
    """
    Airflow task: clean all tables from raw → processed.
    """
    logger.info("Starting cleaning task...")

    manager = CleaningManager()
    result = manager.run()

    if result["failed_count"] > 0:
        raise RuntimeError(
            f"Cleaning failed for {result['failed_count']} table(s). "
            f"Check logs for details."
        )

    logger.info("Cleaning task completed successfully")
    return result
