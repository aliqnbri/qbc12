# core/database.py
"""
Database connection management and session handling.
Provides both sync and async SQLAlchemy engines with connection pooling.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, contextmanager, closing
from typing import Any
import psycopg2
import polars as pl
from loguru import logger
from sqlalchemy import QueuePool, create_engine, event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, declared_attr
from sqlmodel import SQLModel
from io import StringIO
import time
from app.core.config import settings

# ============================================================================
# Base ORM Model
# ============================================================================

class Base(DeclarativeBase):
    """
    Base class for all ORM models.
    Provides common configuration and utilities.
    """
    
    @declared_attr.directive
    def __tablename__(cls) -> str:
        """Auto-generate table name from class name (snake_case)."""
        import re
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()
        return name
    
    def to_dict(self) -> dict[str, Any]:
        """Convert model instance to dictionary."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


# ============================================================================
# Database Engines
# ============================================================================

# Async engine for FastAPI / async operations
async_engine: AsyncEngine = create_async_engine(
    settings.async_database_url,
    echo=settings.postgres_echo,
    # poolclass=QueuePool,
    pool_size=settings.postgres_pool_size,
    max_overflow=settings.postgres_max_overflow,
    pool_pre_ping=settings.postgres_pool_pre_ping,
    pool_recycle=3600,  # Recycle connections after 1 hour
    connect_args={
        "server_settings": {
            "application_name": settings.app_name,
            "jit": "off",  # Disable JIT for faster connection
        },
        "command_timeout": 60,
        "timeout": 30,
    },
)

# Sync engine for blocking operations (Airflow, scripts)
sync_engine = create_engine(
    settings.database_url,  # sync URL (postgresql://)
    echo=settings.postgres_echo,
    # poolclass=QueuePool,
    pool_size=settings.postgres_pool_size,
    max_overflow=settings.postgres_max_overflow,
    pool_pre_ping=settings.postgres_pool_pre_ping,
    pool_recycle=3600,
    connect_args={
        "application_name": settings.app_name,
    },
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ============================================================================
# Connection Event Listeners
# ============================================================================

@event.listens_for(async_engine.sync_engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Log new database connections."""
    logger.debug(f"New DB connection established: {id(dbapi_conn)}")


@event.listens_for(async_engine.sync_engine, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    """Log connection checkouts from pool."""
    logger.debug(f"Connection checked out from pool: {id(dbapi_conn)}")


# ============================================================================
# Session Management
# ============================================================================

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency injection for FastAPI routes.
    Provides async database session with automatic cleanup.
    
    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions in non-FastAPI code.
    
    Usage:
        async with get_db_context() as db:
            result = await db.execute(select(Item))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ============================================================================
# Schema & Table Management
# ============================================================================

def create_schemas_and_tables() -> None:
    """
    Create schemas (raw, processed) and all SQLModel tables.
    
    Uses CREATE SCHEMA IF NOT EXISTS and SQLModel.metadata.create_all.
    Idempotent: safe to run multiple times.
    """
    logger.info("Initializing database schemas and tables...")
    
    with sync_engine.begin() as conn:
        # Create schemas
        for schema in [
            settings.postgres_schema_raw,
            settings.postgres_schema_processed,
            settings.postgres_schema_predictions,
        ]:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))

        logger.info("✓ Schemas created: raw, processed")
        
        # Create all tables from SQLModel metadata
        SQLModel.metadata.create_all(sync_engine)
        
        
        logger.info("✓ Indexes created")


# ============================================================================
# Health Check & Shutdown
# ============================================================================

async def healthcheck_db() -> bool:
    """
    Check database connectivity and readiness.
    Returns True if database is accessible, False otherwise.
    """
    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


async def close_db_connections() -> None:
    """
    Close all database connections.
    Called during application shutdown.
    """
    logger.info("Closing database connections...")
    await async_engine.dispose()
    await sync_engine.dispose()  # ✅ Also dispose sync engine
    logger.info("✓ Database connections closed")


# ============================================================================
# Bulk Insert Utilities
# ============================================================================



def truncate_table(table_name: str, schema: str = "raw") -> None:
    """
    Truncate table for fresh load.
    ⚠️ WARNING: This deletes all data. Use with caution.
    """
    logger.warning(f"Truncating {schema}.{table_name}...")
    with sync_engine.begin() as conn:
        conn.execute(text(f'TRUNCATE TABLE "{schema}"."{table_name}" CASCADE'))
    logger.info(f"✓ Table {schema}.{table_name} truncated")




def _build_sync_url() -> str:
    """Build PostgreSQL connection URL from settings."""
    return settings.database_url


@contextmanager
def get_sync_conn(retries: int = 3, backoff_factor: float = 1.0,autocommit: bool = False):
    """Context manager with exponential backoff retry for transient failures.
    
    Yields a psycopg2 connection. Automatically rolls back on exception.
    """
    conn = None
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg2.connect(_build_sync_url())
            conn.autocommit = autocommit
            yield conn
            return  # Success — skip finally block
        except (psycopg2.OperationalError, psycopg2.extensions.TransactionRollbackError) as e:
            if attempt < retries:
                wait = backoff_factor * (2 ** (attempt - 1))
                logger.warning(f"  ⏳ Connection attempt {attempt}/{retries} failed, retrying in {wait}s...")
                import time
                time.sleep(wait)
            else:
                raise
        finally:
            if conn:
                conn.close()


# ── PostgreSQL COPY ────────────────────────────────────────────────────────

def _pg_copy(
    df: pl.DataFrame,
    schema: str,
    table: str,
    conn,
    truncate: bool = False,
) -> int:
    """High-performance bulk insert via PostgreSQL COPY.
    
    Uses COPY FROM STDIN with CSV format for maximum throughput (~10x faster
    than INSERT statements).
    
    Returns:
        Number of rows inserted.
    """
    buf = StringIO()
    df.write_csv(
        buf,
        null_value="",
        datetime_format="%Y-%m-%d %H:%M:%S",
        date_format="%Y-%m-%d",
    )
    buf.seek(0)

    quoted_cols = ", ".join(f'"{c}"' for c in df.columns)
    copy_sql = (
        f'COPY "{schema}"."{table}" ({quoted_cols}) '
        f"FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')"
    )

    with closing(conn.cursor()) as cur:
        try:
            if truncate:
                cur.execute(f'TRUNCATE TABLE "{schema}"."{table}" RESTART IDENTITY CASCADE')
                logger.debug(f"  Truncated {schema}.{table}")

            cur.copy_expert(copy_sql, buf)
            conn.commit()

            rows = len(df)
            logger.info(f"  ✓ {rows:,} rows → {schema}.{table}")
            return rows

        except Exception as e:
            conn.rollback()
            logger.exception(f"  ✗ Insert failed into {schema}.{table}: {e}")
            raise