# app/api/routes/transform.py
"""
Transform (Feature Engineering) API endpoints.

Provides:
- Build features from processed tables
- List available feature versions
- Get feature metadata
- Delete feature tables
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import polars as pl
from fastapi import APIRouter, Form, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.pipelines.features.builder import FeatureBuilder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/transform", tags=["Transform"])


# ─────────────────────────────────────────────────────────────────────────────
# Response Models
# ─────────────────────────────────────────────────────────────────────────────

class BuildFeaturesResponse(BaseModel):
    """پاسخ feature engineering."""
    status: str = "success"
    task: str = "build_features"
    source_schema: str
    target_schema: str
    target_table: str
    feature_version: str
    row_count: int = Field(..., description="تعداد سطرهای feature table")
    column_count: int = Field(..., description="تعداد ستون‌های feature table")
    execution_time_seconds: float
    completed_at: str
    message: str | None = None


class FeatureTableInfo(BaseModel):
    """اطلاعات یک feature table."""
    schema_name: str
    table_name: str
    row_count: int | None = None
    column_count: int | None = None
    created_at: str | None = None


class FeatureListResponse(BaseModel):
    """لیست feature tables موجود."""
    schema_name: str
    tables: list[FeatureTableInfo]
    count: int


class DeleteFeatureResponse(BaseModel):
    """پاسخ حذف feature table."""
    status: str
    schema_name: str
    table_name: str
    message: str


class HealthCheckResponse(BaseModel):
    """وضعیت سلامت transform module."""
    status: str
    database_url: str
    source_schema: str
    target_schema: str


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Database Operations
# ─────────────────────────────────────────────────────────────────────────────

def get_db_uri() -> str:
    """دریافت connection string برای PostgreSQL."""
    user = settings.postgres_user
    password = settings.postgres_password
    host = settings.postgres_host
    port = settings.postgres_port
    db = settings.postgres_db
    
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def load_tables_from_schema(schema: str, table_names: list[str]) -> dict[str, pl.LazyFrame]:
    """
    بارگذاری جداول از یک schema مشخص.
    
    Args:
        schema: نام schema
        table_names: لیست نام جداول
    
    Returns:
        دیکشنری از LazyFrame ها
    """
    db_uri = get_db_uri()
    tables: dict[str, pl.LazyFrame] = {}
    
    for table_name in table_names:
        try:
            query = f'SELECT * FROM "{schema}"."{table_name}"'
            lf = pl.read_database_uri(query=query, uri=db_uri).lazy()
            tables[table_name] = lf
            logger.info("Loaded table %s.%s", schema, table_name)
        except Exception as e:
            logger.warning("Failed to load table %s.%s: %s", schema, table_name, e)
            # در صورت نیاز می‌توان exception پرتاب کرد
    
    if not tables:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No tables found in schema '{schema}'",
        )
    
    return tables


def write_feature_table(
    df: pl.DataFrame,
    schema: str,
    table_name: str,
    if_exists: str = "replace",
) -> None:
    """
    نوشتن feature table در PostgreSQL.
    
    Args:
        df: Polars DataFrame
        schema: نام schema
        table_name: نام جدول
        if_exists: "replace" یا "append" یا "fail"
    """
    db_uri = get_db_uri()
    
    try:
        df.write_database(
            table_name=table_name,
            connection=db_uri,
            if_table_exists=if_exists,
            engine="sqlalchemy",
        )
        logger.info(
            "Wrote %d rows to %s.%s (if_exists=%s)",
            len(df),
            schema,
            table_name,
            if_exists,
        )
    except Exception as e:
        logger.exception("Failed to write feature table")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write feature table: {str(e)}",
        ) from e


def get_table_info(schema: str, table_name: str) -> dict[str, Any]:
    """
    دریافت اطلاعات یک جدول از information_schema.
    
    Returns:
        دیکشنری حاوی row_count و column_count
    """
    db_uri = get_db_uri()
    
    try:
        # تعداد سطرها
        count_query = f'SELECT COUNT(*) as cnt FROM "{schema}"."{table_name}"'
        row_count_df = pl.read_database_uri(query=count_query, uri=db_uri)
        row_count = int(row_count_df["cnt"][0])
        
        # تعداد ستون‌ها
        col_query = f"""
        SELECT COUNT(*) as cnt
        FROM information_schema.columns
        WHERE table_schema = '{schema}' AND table_name = '{table_name}'
        """
        col_count_df = pl.read_database_uri(query=col_query, uri=db_uri)
        column_count = int(col_count_df["cnt"][0])
        
        return {
            "row_count": row_count,
            "column_count": column_count,
        }
    except Exception as e:
        logger.warning("Could not get table info for %s.%s: %s", schema, table_name, e)
        return {
            "row_count": None,
            "column_count": None,
        }


def list_tables_in_schema(schema: str) -> list[str]:
    """لیست جداول موجود در یک schema."""
    db_uri = get_db_uri()
    
    try:
        query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = '{schema}' AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
        df = pl.read_database_uri(query=query, uri=db_uri)
        return df["table_name"].to_list()
    except Exception as e:
        logger.warning("Could not list tables in schema %s: %s", schema, e)
        return []


def drop_table(schema: str, table_name: str) -> None:
    """حذف یک جدول."""
    db_uri = get_db_uri()
    
    try:
        # استفاده از sqlalchemy برای execute
        from sqlalchemy import create_engine, text
        
        engine = create_engine(db_uri)
        with engine.connect() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "{schema}"."{table_name}" CASCADE'))
            conn.commit()
        
        logger.info("Dropped table %s.%s", schema, table_name)
    except Exception as e:
        logger.exception("Failed to drop table")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to drop table: {str(e)}",
        ) from e


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/build-features",
    response_model=BuildFeaturesResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Build feature table from processed data",
    description="Run feature engineering pipeline: read processed tables, compute features, write to features schema.",
)
def build_features(
    source_schema: str = Form("processed", description="Schema حاوی جداول processed"),
    target_schema: str = Form("features", description="Schema مقصد برای feature table"),
    target_table: str = Form("order_features", description="نام جدول feature"),
    feature_version: str = Form("v1", description="نسخه feature engineering logic"),
    if_exists: str = Form(
        "replace",
        regex="^(replace|append|fail)$",
        description="رفتار در صورت وجود جدول مقصد",
    ),
) -> BuildFeaturesResponse:
    task_start = time.time()

    logger.info(
        "Starting feature engineering: %s.* → %s.%s (version=%s)",
        source_schema,
        target_schema,
        target_table,
        feature_version,
    )

    # لیست جداول مورد نیاز
    required_tables = [
        "orders",
        "order_items",
        "customers",
        "products",
        "sellers",
        "order_payments",
        "order_reviews",
    ]

    try:
        # بارگذاری جداول
        logger.info("Loading tables from schema '%s'", source_schema)
        tables = load_tables_from_schema(source_schema, required_tables)
        
        # بررسی جداول گمشده
        missing = set(required_tables) - set(tables.keys())
        if missing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Missing required tables in schema '{source_schema}': {sorted(missing)}",
            )
        
        # ایجاد FeatureBuilder
        logger.info("Initializing FeatureBuilder (version=%s)", feature_version)
        builder = FeatureBuilder(feature_version=feature_version)
        
        # اجرای feature engineering
        logger.info("Running feature engineering pipeline")
        features_lf = builder.engineer_features(tables)
        
        # جمع‌آوری به DataFrame
        logger.info("Collecting LazyFrame to DataFrame")
        features_df = features_lf.collect()
        
        row_count = len(features_df)
        column_count = len(features_df.columns)
        
        logger.info(
            "Feature table built: %d rows × %d columns",
            row_count,
            column_count,
        )
        
        # نوشتن در PostgreSQL
        logger.info("Writing feature table to %s.%s", target_schema, target_table)
        write_feature_table(
            df=features_df,
            schema=target_schema,
            table_name=target_table,
            if_exists=if_exists,
        )
        
        task_end = time.time()
        execution_time = task_end - task_start
        
        logger.info(
            "Feature engineering completed in %.2f seconds",
            execution_time,
        )
        
        return BuildFeaturesResponse(
            source_schema=source_schema,
            target_schema=target_schema,
            target_table=target_table,
            feature_version=feature_version,
            row_count=row_count,
            column_count=column_count,
            execution_time_seconds=round(execution_time, 2),
            completed_at=datetime.utcnow().isoformat(),
            message=f"Feature table created with {row_count:,} rows",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Feature engineering failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature engineering failed: {str(e)}",
        ) from e
    @router.get("/features",
    response_model=FeatureListResponse,
    summary="List feature tables",
    description="List all feature tables in the features schema.",
    )
    def list_feature_tables(

        schema: str = Query("features", description="Schema name"),
        ) -> FeatureListResponse:
        logger.info("Listing tables in schema '%s'", schema)

try:
    table_names = list_tables_in_schema(schema)
    
    tables_info: list[FeatureTableInfo] = []
    
    for table_name in table_names:
        info = get_table_info(schema, table_name)
        tables_info.append(
            FeatureTableInfo(
                schema_name=schema,
                table_name=table_name,
                row_count=info.get("row_count"),
                column_count=info.get("column_count"),
                created_at=None,  # می‌توان از pg_class استخراج کرد
            )
        )
    
    return FeatureListResponse(
        schema_name=schema,
        tables=tables_info,
        count=len(tables_info),
    )

except Exception as e:
    logger.exception("Failed to list feature tables")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Failed to list feature tables: {str(e)}",
    ) from e
@router.get(

"/features/{table_name}",
response_model=FeatureTableInfo,
summary="Get feature table metadata",
description="Get detailed metadata for a specific feature table.",
)

def get_feature_table_metadata(
    table_name: str,
    schema: str = Query("features", description="Schema name"),
    ) -> FeatureTableInfo:
    logger.info("Getting metadata for %s.%s", schema, table_name)

# بررسی وجود جدول
tables = list_tables_in_schema(schema)
if table_name not in tables:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Table '{table_name}' not found in schema '{schema}'",
    )

try:
    info = get_table_info(schema, table_name)
    
    return FeatureTableInfo(
        schema_name=schema,
        table_name=table_name,
        row_count=info.get("row_count"),
        column_count=info.get("column_count"),
        created_at=None,
    )

except Exception as e:
    logger.exception("Failed to get table metadata")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Failed to get table metadata: {str(e)}",
    ) from e
@router.delete(

"/features/{table_name}",
response_model=DeleteFeatureResponse,
summary="Delete feature table",
description="Drop a feature table from the database.",
)

def delete_feature_table(

    table_name: str,
    schema: str = Query("features", description="Schema name"),
    ) -> DeleteFeatureResponse:
    logger.warning("Deleting table %s.%s", schema, table_name)

# بررسی وجود جدول
tables = list_tables_in_schema(schema)
if table_name not in tables:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Table '{table_name}' not found in schema '{schema}'",
    )

try:
    drop_table(schema, table_name)
    
    return DeleteFeatureResponse(
        status="deleted",
        schema_name=schema,
        table_name=table_name,
        message=f"Table '{schema}.{table_name}' deleted successfully",
    )

except HTTPException:
    raise
except Exception as e:
    logger.exception("Failed to delete table")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Failed to delete table: {str(e)}",
    ) from e
@router.get(

"/health",
response_model=HealthCheckResponse,
summary="Transform module health check",
description="Check if transform module can connect to database.",
)

def transform_health_check() -> HealthCheckResponse:
    db_uri = get_db_uri()

# بررسی اتصال
try:
    test_query = "SELECT 1 as test"
    pl.read_database_uri(query=test_query, uri=db_uri)
    status_msg = "healthy"
except Exception as e:
    logger.error("Database connection failed: %s", e)
    status_msg = "unhealthy"

return HealthCheckResponse(
    status=status_msg,
    database_url=db_uri.split("@")[-1],  # فقط host:port/db
    source_schema="processed",
    target_schema="features",
)
