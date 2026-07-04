"""SQLAlchemy engine and session factories."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

RAW_SCHEMA = "raw"
PROCESSED_SCHEMA = "processed"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the cached SQLAlchemy engine for the application database."""
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        future=True,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Return the cached session factory bound to the engine."""
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional session scope with commit/rollback semantics."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_schemas(engine: Engine | None = None) -> None:
    """Create the ``raw`` and ``processed`` schemas if absent."""
    engine = engine or get_engine()
    with engine.begin() as connection:
        for schema in (RAW_SCHEMA, PROCESSED_SCHEMA):
            connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
