# File: app/airflow/tasks/export_dataset.py
"""
Airflow task: Export feature table from PostgreSQL to Parquet for model training.
Optionally tracks the output file with DVC.
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import polars as pl
from loguru import logger

from app.core.config import settings
from app.core.database import get_sync_conn

# ─── Output path ──────────────────────────────
OUTPUT_PATH = settings.data_dir / "model_dataset.parquet"

# ─── Source table ─────────────────────────────
FEATURE_SCHEMA = "features"
FEATURE_TABLE  = "order_features"


# ─── Public task entrypoint ───────────────────
def run_export_dataset_task(
    use_dvc: bool = True,
    compression: str = "snappy",
) -> dict:
    """
    Airflow task: Read features.order_features → write data/model_dataset.parquet.

    Args:
        use_dvc: Run `dvc add` after writing the file (default: True)
        compression: Parquet compression codec (default: "snappy")

    Returns:
        {"status": "success", "row_count": int, "path": str, "columns": list}
    """
    task_start = datetime.now()
    logger.info(f"📤 Starting dataset export from {FEATURE_SCHEMA}.{FEATURE_TABLE}")

    # Step 1: Read from database
    df = _read_features()

    if df.is_empty():
        raise ValueError(
            f"Table {FEATURE_SCHEMA}.{FEATURE_TABLE} is empty. "
            "Run build_features task first."
        )

    logger.info(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Step 2: Write Parquet
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUTPUT_PATH, compression=compression)
    logger.success(f"💾 Written: {OUTPUT_PATH}")

    # Step 3: DVC tracking (optional)
    if use_dvc:
        _dvc_add(OUTPUT_PATH)

    elapsed = (datetime.now() - task_start).total_seconds()

    result = {
        "status": "success",
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns,
        "path": str(OUTPUT_PATH),
        "compression": compression,
        "execution_time_seconds": round(elapsed, 2),
    }
    logger.info(f"🎉 Export complete: {result}")
    return result


# ─── Helpers ──────────────────────────────────
def _read_features() -> pl.DataFrame:
    """Read order_features table using sync connection."""
    query = f'SELECT * FROM "{FEATURE_SCHEMA}"."{FEATURE_TABLE}"'
    with get_sync_conn() as conn:
        df = pl.read_database(query, conn)
    return df


def _dvc_add(path: Path) -> None:
    """Run `dvc add <path>` from data directory. Logs warning on failure."""
    logger.info(f"📌 DVC tracking: {path}")
    try:
        # Run from the data directory parent (project root)
        cwd = settings.data_dir.parent
        result = subprocess.run(
            ["dvc", "add", str(path.relative_to(cwd))],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.success(f"✅ DVC: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️  DVC add failed (non-fatal): {e.stderr.strip()}")
    except FileNotFoundError:
        logger.warning("⚠️  DVC not installed — skipping version tracking")
