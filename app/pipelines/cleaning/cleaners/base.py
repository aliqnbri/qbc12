"""
Base cleaner class for all table-specific cleaners.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import polars as pl
from sqlalchemy import text

from app.core.database import get_sync_conn, _pg_copy


class BaseCleaner(ABC):
    """Base class for cleaning a single table from raw → processed."""

    def __init__(self, table_name: str):
        self.table_name = table_name
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.raw_schema = "raw"
        self.processed_schema = "processed"

    @abstractmethod
    def clean(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Apply cleaning transformations.
        Must be implemented by each table cleaner.
        """
        pass

    def load_from_raw(self) -> pl.LazyFrame:
        """Load table from raw schema using psycopg2."""
        query = f'SELECT * FROM {self.raw_schema}."{self.table_name}"'
        self.logger.info(f"Loading {self.table_name} from raw schema...")
        
        with get_sync_conn() as conn:
            df = pl.read_database(query, connection=conn)
        
        self.logger.info(f"Loaded {len(df):,} rows from raw.{self.table_name}")
        return df.lazy()

    def write_to_processed(self, lf: pl.LazyFrame) -> int:
        """Write cleaned data to processed schema using COPY."""
        df = lf.collect()
        row_count = len(df)

        self.logger.info(f"Writing {row_count:,} rows to processed.{self.table_name}...")

        with get_sync_conn() as conn:
            with conn.cursor() as cur:
                # Create schema if not exists
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.processed_schema}")
                
                # Drop and recreate table
                cur.execute(f'DROP TABLE IF EXISTS "{self.processed_schema}"."{self.table_name}" CASCADE')
                
                # Create table from DataFrame schema
                self._create_table_from_schema(df, cur)
                
                conn.commit()

            # Bulk insert using _pg_copy
            _pg_copy(
                df=df,
                schema=self.processed_schema,
                table=self.table_name,
                conn=conn,
                truncate=False,  # Already dropped above
            )

        self.logger.info(f"✓ {self.table_name}: {row_count:,} rows written to processed")
        return row_count

    def _create_table_from_schema(self, df: pl.DataFrame, cursor) -> None:
        """Create table based on Polars DataFrame schema."""
        type_mapping = {
            pl.Int64: "BIGINT",
            pl.Int32: "INTEGER",
            pl.Float64: "DOUBLE PRECISION",
            pl.Float32: "REAL",
            pl.Utf8: "TEXT",
            pl.Boolean: "BOOLEAN",
            pl.Date: "DATE",
            pl.Datetime: "TIMESTAMP",
        }

        columns = []
        for col_name, dtype in zip(df.columns, df.dtypes):
            pg_type = type_mapping.get(dtype, "TEXT")
            columns.append(f'"{col_name}" {pg_type}')

        create_sql = (
            f'CREATE TABLE "{self.processed_schema}"."{self.table_name}" '
            f"({', '.join(columns)})"
        )
        cursor.execute(create_sql)

    def run(self) -> dict[str, Any]:
        """Execute full cleaning pipeline for this table."""
        self.logger.info(f"Starting cleaning for {self.table_name}...")
        
        try:
            # Load
            lf = self.load_from_raw()
            initial_count = lf.select(pl.len()).collect().item()

            # Clean
            lf_clean = self.clean(lf)
            
            # Write
            final_count = self.write_to_processed(lf_clean)

            removed = initial_count - final_count
            self.logger.info(
                f"✓ {self.table_name}: {initial_count:,} → {final_count:,} "
                f"({removed:,} rows removed)"
            )

            return {
                "table": self.table_name,
                "status": "success",
                "initial_rows": initial_count,
                "final_rows": final_count,
                "removed_rows": removed,
            }

        except Exception as e:
            self.logger.error(f"✗ {self.table_name} cleaning failed: {e}")
            return {
                "table": self.table_name,
                "status": "failed",
                "error": str(e),
            }
