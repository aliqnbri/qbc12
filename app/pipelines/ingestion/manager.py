"""Orchestrates the complete raw data ingestion pipeline.

Discovers CSV files automatically, matches them to table definitions,
and executes loading in dependency-respecting order.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.database import create_schemas_and_tables
from app.pipelines.ingestion.definitions import TableRegistry
from app.pipelines.ingestion.loader import TableLoader
from app.utils.utils import discover_csv_files

logger = logging.getLogger(__name__)


class IngestionManager:
    """High-level manager for the complete ingestion pipeline.
    
    Workflow:
        1. Discover CSV files in data directory
        2. Match to table definitions
        3. Create database schemas
        4. Load tables in dependency order
        5. Report execution metrics
    
    Attributes:
        data_dir: Root directory containing CSV files
        truncate_before_load: Whether to truncate tables before loading
        fail_fast: If True, abort pipeline on first table failure
    """
    
    def __init__(
        self,
        data_dir: Path | None = None,
        truncate_before_load: bool = True,
        fail_fast: bool = True,
    ) -> None:
        self.data_dir = (data_dir or settings.raw_data_dir).expanduser().resolve()
        self.truncate_before_load = truncate_before_load
        self.fail_fast = fail_fast
        
        # Validate data directory existence
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}"
            )
    
    def run(self, validate_only: bool = False) -> dict[str, Any]:
        """Execute the complete ingestion pipeline.
        
        Args:
            validate_only: If True, validate CSVs without database insertion
        
        Returns:
            Execution summary containing:
                - status: 'completed' | 'completed_with_errors' | 'validated'
                - total_files: Number of CSV files discovered
                - success_count: Tables loaded successfully
                - failed_count: Tables that failed
                - skipped_count: Tables skipped (no definition or missing CSV)
                - total_rows_inserted: Aggregate row count
                - failed_tables: List of failed table names
                - reports: Per-table detailed reports
                - data_dir: Source directory path
        
        Raises:
            RuntimeError: If fail_fast=True and any table fails
        """
        logger.info("=" * 80)
        logger.info("▶ Starting Raw Data Ingestion Pipeline")
        logger.info("  Data directory: %s", self.data_dir)
        logger.info("  Mode: %s", "validate-only" if validate_only else "ingest")
        logger.info("  Truncate before load: %s", self.truncate_before_load)
        logger.info("  Fail fast: %s", self.fail_fast)
        logger.info("=" * 80)
        
        # Step 1: Discover CSV files
        logger.info("Step 1: Discovering CSV files...")
        csv_files = discover_csv_files(self.data_dir)
        logger.info("  Found %d CSV files", len(csv_files))
        
        # Step 2: Create schemas and tables (skip in validate-only mode)
        if not validate_only:
            logger.info("Step 2: Creating database schemas and tables...")
            create_schemas_and_tables()
        
        # Step 3: Load tables in dependency order
        logger.info("Step 3: Processing tables in dependency order...")
        loader = TableLoader(truncate_before_load=self.truncate_before_load)
        
        reports: list[dict[str, Any]] = []
        failed_tables: list[str] = []
        
        for definition in TableRegistry.get_load_order():
            # Find matching CSV file
            csv_path = csv_files.get(definition.table)
            
            if not csv_path:
                logger.warning(
                    "  ⊘ No CSV found for table '%s' (expected pattern matching '%s')",
                    definition.table,
                    definition.csv_pattern or f"*{definition.table}*.csv",
                )
                reports.append({
                    "table": definition.full_table_name,
                    "status": "skipped",
                    "reason": "no_matching_csv",
                })
                continue
            
            # Load table
            report = loader.load(csv_path, definition, validate_only=validate_only)
            reports.append(report)
            
            # Handle failure
            if report["status"] == "failed":
                failed_tables.append(definition.table)
                if self.fail_fast:
                    logger.error("  Aborting due to fail_fast=True")
                    break
        
        # Step 4: Aggregate metrics
        total_rows = sum(
            report.get("rows_inserted", 0)
            for report in reports
            if report["status"] == "success"
        )
        
        success_count = sum(
            1 for report in reports if report["status"] == "success"
        )
        failed_count = sum(
            1 for report in reports if report["status"] == "failed"
        )
        skipped_count = sum(
            1 for report in reports if report["status"] == "skipped"
        )
        validated_count = sum(
            1 for report in reports if report["status"] == "validated"
        )
        
        # Step 5: Log summary
        logger.info("=" * 80)
        logger.info("Pipeline Execution Summary")
        logger.info("=" * 80)
        logger.info("  Total CSV files discovered: %d", len(csv_files))
        logger.info("  Total tables processed: %d", len(reports))
        logger.info("  ✓ Success: %d", success_count)
        logger.info("  ✗ Failed: %d", failed_count)
        logger.info("  ⊘ Skipped: %d", skipped_count)
        logger.info("  ✓ Validated: %d", validated_count)
        logger.info("  Total rows inserted: %d", total_rows)
        
        if failed_tables:
            logger.warning("  Failed tables: %s", ", ".join(failed_tables))
        else:
            logger.info("  ✓ All tables processed successfully!")
        
        logger.info("=" * 80)
        
        # Determine final status
        if validate_only and not failed_tables:
            final_status = "validated"
        elif failed_tables:
            final_status = "completed_with_errors"
        else:
            final_status = "completed"
        
        return {
            "status": final_status,
            "total_files": len(csv_files),
            "success_count": success_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "validated_count": validated_count,
            "total_rows_inserted": total_rows,
            "failed_tables": failed_tables,
            "reports": reports,
            "data_dir": str(self.data_dir),
        }
