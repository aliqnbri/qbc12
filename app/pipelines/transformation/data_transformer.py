# File: app/pipelines/transformation/data_transformer.py
"""
Data transformation pipeline: raw CSV → schema-compliant processed data.
"""
import logging
from typing import Any

import polars as pl

from app.pipelines.transformation.schema_registry import ColumnSpec, get_table_schema

logger = logging.getLogger(__name__)


class DataTransformationError(Exception):
    """Raised when data transformation or validation fails."""
    pass


def rename_csv_aliases(df: pl.DataFrame, table_name: str) -> pl.DataFrame:
    """
    Rename known CSV column typos/aliases to canonical schema names.
    
    dataset has a typo in products: "lenght" → "length"
    """
    rename_map = {}
    
    if table_name == "products":
        if "product_name_lenght" in df.columns:
            rename_map["product_name_lenght"] = "product_name_length"
        if "product_description_lenght" in df.columns:
            rename_map["product_description_lenght"] = "product_description_length"
    
    if rename_map:
        df = df.rename(rename_map)
        logger.info(f"Renamed CSV columns in {table_name}: {rename_map}")
    
    return df


def detect_and_parse_timestamps(df: pl.DataFrame) -> pl.DataFrame:
    """
    Auto-detect and parse timestamp columns from strings.
    
    Looks for columns with names containing:
    - 'timestamp'
    - '_date'
    - '_at'
    - 'won_date'
    - 'first_contact'
    """
    timestamp_patterns = ["timestamp", "_date", "_at", "won_date", "first_contact"]
    parsed_count = 0
    
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            if any(pattern in col.lower() for pattern in timestamp_patterns):
                try:
                    cleaned = pl.col(col).str.strip_chars()
                    # Try full datetime, then fall back to date-only
                    df = df.with_columns(
                        pl.coalesce(
                            cleaned.str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
                            cleaned.str.strptime(pl.Date, "%Y-%m-%d", strict=False).cast(pl.Datetime),
                        ).alias(col)
                    )
                    parsed_count += 1
                except Exception:
                    logger.warning(f"Could not parse timestamp column {col}")
    
    if parsed_count > 0:
        logger.info(f"Parsed {parsed_count} timestamp columns")
    
    return df


def clean_string_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize string columns: strip whitespace, lowercase categorical fields.
    """
    categorical_cols = {
        "order_status", "payment_type", "origin", "customer_state", "seller_state",
        "business_segment", "lead_type", "lead_behaviour_profile", "business_type",
        "product_category_name"
    }
    
    preserve_case_patterns = ["_id", "mql", "sdr", "sr", "review_comment", "landing_page"]
    
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            # Strip and null empty strings
            df = df.with_columns(
                pl.col(col)
                .str.strip_chars()
                .replace("", None)
                .alias(col)
            )
            
            # Lowercase categorical unless preserve pattern matches
            should_preserve = any(pattern in col.lower() for pattern in preserve_case_patterns)
            if col in categorical_cols and not should_preserve:
                df = df.with_columns(pl.col(col).str.to_lowercase().alias(col))
    
    return df


def apply_schema_transformations(df: pl.DataFrame, table_name: str) -> pl.DataFrame:
    """
    Apply schema-driven casting and missing-column handling.
    
    Raises:
        DataTransformationError: if non-nullable column is missing or contains nulls
    """
    schema = get_table_schema(table_name)
    
    # Handle missing columns
    for col_name, spec in schema.items():
        if col_name not in df.columns:
            if spec.nullable:
                default_val = spec.default if spec.default is not None else None
                df = df.with_columns(pl.lit(default_val).alias(col_name))
                logger.warning(f"Added missing column {col_name} with default {default_val}")
            else:
                raise DataTransformationError(
                    f"Non-nullable column {col_name} missing in table {table_name}"
                )
    
    # Cast columns to expected types
    for col_name, spec in schema.items():
        if col_name not in df.columns:
            continue
        
        try:
            target_dtype = spec.dtype
            current_dtype = df[col_name].dtype
            
            # --- Datetime ---
            if target_dtype == pl.Datetime:
                if current_dtype == pl.Utf8:
                    # Only parse from string
                    df = df.with_columns(
                        pl.coalesce(
                            pl.col(col_name).str.strptime(
                                pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False
                            ),
                            pl.col(col_name).str.strptime(
                                pl.Date, "%Y-%m-%d", strict=False
                            ).cast(pl.Datetime),
                        )
                    )
                elif current_dtype == pl.Date:
                    # Cast Date → Datetime
                    df = df.with_columns(pl.col(col_name).cast(pl.Datetime))
                elif current_dtype != pl.Datetime:
                    # Attempt generic cast
                    df = df.with_columns(pl.col(col_name).cast(pl.Datetime, strict=False))
                # else: already Datetime, skip
            
            # --- Date ---
            elif target_dtype == pl.Date:
                if current_dtype == pl.Utf8:
                    # Only parse from string
                    df = df.with_columns(
                        pl.col(col_name).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                    )
                elif current_dtype == pl.Datetime:
                    # Cast Datetime → Date
                    df = df.with_columns(pl.col(col_name).cast(pl.Date))
                elif current_dtype != pl.Date:
                    # Attempt generic cast
                    df = df.with_columns(pl.col(col_name).cast(pl.Date, strict=False))
                # else: already Date, skip
            
            # --- Float ---
            elif target_dtype == pl.Float64:
                df = df.with_columns(pl.col(col_name).cast(pl.Float64, strict=False))
            
            # --- Boolean ---
            elif target_dtype == pl.Boolean:
                if current_dtype == pl.Utf8:
                    df = df.with_columns(
                        pl.col(col_name)
                        .str.to_lowercase()
                        .replace({
                            "true": True, "1": True, "yes": True,
                            "false": False, "0": False, "no": False
                        })
                        .cast(pl.Boolean, strict=False)
                    )
                else:
                    df = df.with_columns(pl.col(col_name).cast(pl.Boolean, strict=False))
            
            # --- Integer with range validation ---
            elif target_dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
                temp_col = df[col_name].cast(pl.Int64, strict=False)
                
                if target_dtype == pl.Int8:
                    if (temp_col.min() < -128) or (temp_col.max() > 127):
                        raise DataTransformationError(f"Column {col_name} out of Int8 range")
                elif target_dtype == pl.Int16:
                    if (temp_col.min() < -32768) or (temp_col.max() > 32767):
                        raise DataTransformationError(f"Column {col_name} out of Int16 range")
                elif target_dtype == pl.Int32:
                    if (temp_col.min() < -2147483648) or (temp_col.max() > 2147483647):
                        raise DataTransformationError(f"Column {col_name} out of Int32 range")
                
                df = df.with_columns(temp_col.cast(target_dtype, strict=False).alias(col_name))
            
            else:
                df = df.with_columns(pl.col(col_name).cast(target_dtype, strict=False))
        
        except Exception as e:
            logger.error(f"Failed to cast {col_name} to {target_dtype}: {e}")
            raise DataTransformationError(f"Casting error for {col_name}: {e}")
    
    # Validate non-nullable columns
    for col_name, spec in schema.items():
        if not spec.nullable:
            null_count = df[col_name].null_count()
            if null_count > 0:
                raise DataTransformationError(
                    f"Non-nullable column {col_name} contains {null_count} nulls"
                )
    
    # Reorder columns to match schema
    df = df.select([col for col in schema.keys() if col in df.columns])
    
    return df


def validate_data_quality(
    df: pl.DataFrame,
    table_name: str,
    schema: dict[str, ColumnSpec]
) -> dict[str, Any]:
    """
    Produce data quality report.
    
    Returns:
        Quality report dict with warnings and pass/fail status
    """
    report = {
        "table_name": table_name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "null_counts": {},
        "duplicate_count": 0,
        "warnings": [],
        "passed": True
    }
    
    # Null counts
    for col in df.columns:
        null_count = df[col].null_count()
        if null_count > 0:
            null_pct = (null_count / len(df)) * 100
            report["null_counts"][col] = {"count": null_count, "percentage": null_pct}
            
            spec = schema.get(col)
            if spec and not spec.nullable:
                report["warnings"].append(f"Non-nullable column {col} has {null_count} nulls")
                report["passed"] = False
    
    # Duplicate detection
    primary_keys = [col for col, spec in schema.items() if spec.primary_key]
    if primary_keys:
        dup_count = len(df) - df.select(primary_keys).n_unique()
        report["duplicate_count"] = dup_count
        if dup_count > 0:
            report["warnings"].append(f"Found {dup_count} duplicate primary keys")
    else:
        # Fallback heuristic: first non-nullable _id column
        id_cols = [col for col, spec in schema.items() if col.endswith("_id") and not spec.nullable]
        if id_cols:
            dup_count = len(df) - df.select(id_cols[0]).n_unique()
            report["duplicate_count"] = dup_count
    
    return report


def transform_dataframe(
    df: pl.DataFrame,
    table_name: str,
    drop_duplicates: bool = True,
    raise_on_failure: bool = False
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """
    Full transformation pipeline: raw → schema-compliant processed data.
    
    Pipeline:
    1. Detect and parse timestamps
    2. Clean string columns
    3. Apply schema transformations
    4. Optional deduplication
    5. Validate data quality
    
    Returns:
        (transformed_df, quality_report)
    """
    logger.info(f"Transforming table {table_name} with {len(df)} rows")
    
    # Step 1: Parse timestamps
    df = detect_and_parse_timestamps(df)
    
    # Step 2: Clean strings
    df = clean_string_columns(df)
    
    # Step 3: Rename CSV aliases  
    df = rename_csv_aliases(df, table_name)
    
    # Step 4: Apply schema
    df = apply_schema_transformations(df, table_name)
    
    # Step 5: Deduplication
    if drop_duplicates:
        schema = get_table_schema(table_name)
        primary_keys = [col for col, spec in schema.items() if spec.primary_key]
        if primary_keys:
            before = len(df)
            df = df.unique(subset=primary_keys, keep="first")
            after = len(df)
            if before != after:
                logger.info(f"Dropped {before - after} duplicates from {table_name}")
    
    # Step 6: Validate quality
    schema = get_table_schema(table_name)
    quality_report = validate_data_quality(df, table_name, schema)
    
    if not quality_report["passed"]:
        logger.error(f"Quality validation failed for {table_name}: {quality_report['warnings']}")
        if raise_on_failure:
            raise DataTransformationError(f"Quality check failed: {quality_report['warnings']}")
    
    logger.info(f"Transformation complete: {len(df)} rows, {len(df.columns)} columns")
    
    return df, quality_report
