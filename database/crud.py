"""CRUD helpers over the processed schema."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from database.models import OrderFeature, PredictionLog
from ml.feature_engineering import FEATURE_COLUMNS
from utils.logging import get_logger

logger = get_logger(__name__)

_FEATURE_TABLE_COLUMNS = [
    "order_id",
    "order_purchase_timestamp",
    "seller_id",
    "customer_unique_id",
    "late_delivery",
    *FEATURE_COLUMNS,
]


def replace_order_features(session: Session, features: pl.DataFrame, chunk_size: int = 5000) -> int:
    """Replace the ``processed.order_features`` table content.

    Args:
        session: Open SQLAlchemy session (caller controls the transaction).
        features: Frame containing at least the feature-table columns.
        chunk_size: Bulk-insert batch size.

    Returns:
        Number of rows inserted.
    """
    available = [column for column in _FEATURE_TABLE_COLUMNS if column in features.columns]
    rows = features.select(available).to_dicts()
    session.execute(delete(OrderFeature))
    for start in range(0, len(rows), chunk_size):
        session.bulk_insert_mappings(OrderFeature, rows[start : start + chunk_size])
    logger.info("Inserted %d rows into processed.order_features", len(rows))
    return len(rows)


def log_prediction(
    session: Session,
    order_id: str,
    late_probability: float,
    risk_level: str,
    model_name: str,
    model_version: str,
    latency_seconds: float | None = None,
) -> PredictionLog:
    """Persist a served prediction for auditability and drift analysis."""
    record = PredictionLog(
        order_id=order_id,
        late_probability=late_probability,
        risk_level=risk_level,
        model_name=model_name,
        model_version=model_version,
        latency_seconds=latency_seconds,
    )
    session.add(record)
    return record


def get_order_features(session: Session, order_ids: Sequence[str]) -> list[OrderFeature]:
    """Fetch stored feature rows for the given order ids."""
    statement = select(OrderFeature).where(OrderFeature.order_id.in_(list(order_ids)))
    return list(session.execute(statement).scalars())


def count_order_features(session: Session) -> int:
    """Count rows in ``processed.order_features``."""
    return len(list(session.execute(select(OrderFeature.order_id)).scalars()))


def recent_predictions(session: Session, limit: int = 100) -> list[PredictionLog]:
    """Return the most recent served predictions."""
    statement = (
        select(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(limit)
    )
    return list(session.execute(statement).scalars())
