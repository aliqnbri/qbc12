"""Single-table ingestion logic with transformation and validation.

Handles reading CSV, applying schema transformations via schema_registry
and data_transformer, and inserting into PostgreSQL using efficient COPY.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import polars as pl

from app.core.database import get_sync_conn, _pg_copy
from app.pipelines.ingestion.definitions import TableDefinition
from app.pipelines.transformation.data_transformer import transform_dataframe

logger = logging.getLogger(__name__)


class TableLoader:
    """Loads a single CSV file into PostgreSQL with validation.
    
    Workflow:
        1. Read CSV with null handling
        2. Apply schema transformations (type coercion, validation)
        3. Insert via PostgreSQL COPY (fast bulk load)
        4. Return detailed ingestion report
    
    Attributes:
        truncate_before_load: Whether to TRUNCATE table before insert
    """
    
    def __init__(self, truncate_before_load: bool = True) -> None:
        self.truncate_before_load = truncate_before_load
    
    def load(
        self,
        csv_path: Path,
        definition: TableDefinition,
        validate_only: bool = False,
    ) -> dict[str, Any]:
        """Load a single table from CSV to PostgreSQL.
        
        Args:
            csv_path: Path to source CSV file
            definition: Table metadata (schema, table name, PKs)
            validate_only: If True, validate but skip database insertion
        
        Returns:
            Ingestion report containing:
                - csv_file: Source filename
                - table: Full qualified table name
                - status: 'success' | 'failed' | 'skipped' | 'validated'
                - rows_loaded: Count from CSV
                - rows_inserted: Count written to DB (if not validate_only)
                - rows_dropped: Difference (duplicates, invalid rows)
                - quality_report: Detailed validation metrics
                - error: Exception message (if failed)
        """
        if not csv_path.exists():
            logger.warning(
                "File not found: %s → skipping table %s",
                csv_path,
                definition.full_table_name,
            )
            return {
                "csv_file": csv_path.name,
                "table": definition.full_table_name,
                "status": "skipped",
                "reason": "file_not_found",
            }
        
        logger.info(
            "Loading %s → %s",
            csv_path.name,
            definition.full_table_name,
        )
        
        try:
            # Step 1: Read CSV with robust null handling
            df = self._read_csv(csv_path)
            logger.debug(
                "  Read %d rows × %d columns",
                len(df),
                len(df.columns),
            )
            
            # Step 2: Transform and validate using schema_registry
            df_clean, quality_report = transform_dataframe(
                df,
                table_name=definition.table,
                drop_duplicates=True,
            )
            
            # Validate-only mode: skip insertion
            if validate_only:
                logger.info(
                    "  ✓ Validated %d rows (no insert)",
                    len(df_clean),
                )
                return {
                    "csv_file": csv_path.name,
                    "table": definition.full_table_name,
                    "status": "validated",
                    "rows_loaded": len(df),
                    "rows_valid": len(df_clean),
                    "rows_dropped": len(df) - len(df_clean),
                    "quality_report": quality_report,
                }
            
            # Step 3: Insert via PostgreSQL COPY
            rows_inserted = self._insert_dataframe(
                df_clean,
                definition.schema,
                definition.table,
            )
            
            logger.info(
                "  ✓ Inserted %d rows into %s",
                rows_inserted,
                definition.full_table_name,
            )
            
            return {
                "csv_file": csv_path.name,
                "table": definition.full_table_name,
                "status": "success",
                "rows_loaded": len(df),
                "rows_inserted": rows_inserted,
                "rows_dropped": len(df) - rows_inserted,
                "quality_report": quality_report,
            }
        
        except Exception as e:
            logger.error(
                "  ✗ Failed to load %s: %s",
                csv_path.name,
                e,
                exc_info=True,
            )
            return {
                "csv_file": csv_path.name,
                "table": definition.full_table_name,
                "status": "failed",
                "error": str(e),
            }
    
    @staticmethod
    def _read_csv(path: Path) -> pl.DataFrame:
        """Read CSV with standardized null handling and error tolerance.
        
        Args:
            path: Path to CSV file
        
        Returns:
            Polars DataFrame with inferred schema
        """
        return pl.read_csv(
            path,
            infer_schema_length=10_000,
            null_values=["", "NA", "null", "NULL", "N/A", "nan"],
            ignore_errors=True,
        )
    
    def _insert_dataframe(self,df: pl.DataFrame,schema: str,table: str,) -> int:
        """Insert DataFrame into PostgreSQL using COPY protocol.
        
        Args:
            df: Cleaned and validated DataFrame
            schema: Target schema name
            table: Target table name
        
        Returns:
            Number of rows inserted
        """
        
        with get_sync_conn() as conn:
            return _pg_copy(
            df=df,
            schema=schema,
            table=table,
            conn=conn,
            truncate=self.truncate_before_load,
        )