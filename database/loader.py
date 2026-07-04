"""Loaders that move CSV data into the raw and processed PostgreSQL schemas.

CLI:
    python -m database.loader --data-dir data [--with-processed]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from sqlalchemy import Engine, text

from database.models import Base
from database.session import (
    PROCESSED_SCHEMA,
    RAW_SCHEMA,
    create_schemas,
    get_engine,
    session_scope,
)
from database import crud
from ml.cleaning import clean_datasets
from utils.logging import configure_logging, get_logger
from utils.normalize_dataset import load_datasets

logger = get_logger(__name__)

#: Datasets persisted into the raw schema (the engineered ``final`` reference
#: frame is intentionally excluded — it is for validation only).
RAW_TABLES: tuple[str, ...] = (
    "orders",
    "order_items",
    "order_payments",
    "order_reviews",
    "customers",
    "sellers",
    "products",
    "marketing_qualified_leads",
    "closed_deals",
    "product_category_name_translation",
)

#: Source-column typos normalized before persistence.
_COLUMN_RENAMES: dict[str, str] = {
    "product_name_lenght": "product_name_length",
    "product_description_lenght": "product_description_length",
}

_BOOLEAN_COLUMNS: dict[str, list[str]] = {"closed_deals": ["has_company", "has_gtin"]}


def init_database(engine: Engine | None = None) -> None:
    """Create schemas and all ORM-defined tables (idempotent)."""
    engine = engine or get_engine()
    create_schemas(engine)
    Base.metadata.create_all(engine)
    logger.info("Database schemas and tables ensured")


def _normalize_for_db(name: str, frame: pl.DataFrame) -> pl.DataFrame:
    """Rename typo'd columns and coerce booleans before writing to Postgres."""
    renames = {old: new for old, new in _COLUMN_RENAMES.items() if old in frame.columns}
    if renames:
        frame = frame.rename(renames)
    for column in _BOOLEAN_COLUMNS.get(name, []):
        if column in frame.columns and frame.schema[column] == pl.Utf8:
            frame = frame.with_columns(
                pl.col(column)
                .str.to_lowercase()
                .replace_strict(
                    {"true": True, "false": False},
                    default=None,
                    return_dtype=pl.Boolean,
                )
                .alias(column)
            )
    return frame


def _write_table(
    frame: pl.DataFrame, schema: str, table: str, engine: Engine
) -> None:
    """Truncate and bulk-append a frame into an existing table."""
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE TABLE {schema}.{table} CASCADE"))
    frame.write_database(
        table_name=f"{schema}.{table}",
        connection=engine,
        if_table_exists="append",
        engine="sqlalchemy",
    )
    logger.info("Loaded %d rows into %s.%s", frame.height, schema, table)


def load_raw_to_db(
    data_dir: str | Path,
    engine: Engine | None = None,
    datasets: dict[str, pl.DataFrame] | None = None,
) -> dict[str, int]:
    """Load every raw CSV into the ``raw`` schema.

    Returns:
        Mapping of table name to inserted row count.
    """
    engine = engine or get_engine()
    init_database(engine)
    datasets = datasets or load_datasets(data_dir)

    counts: dict[str, int] = {}
    for name in RAW_TABLES:
        if name not in datasets:
            logger.warning("Dataset %s not found in %s; skipping", name, data_dir)
            continue
        frame = _normalize_for_db(name, datasets[name])
        _write_table(frame, RAW_SCHEMA, name, engine)
        counts[name] = frame.height
    return counts


def write_processed_tables(
    cleaned: dict[str, pl.DataFrame], engine: Engine | None = None
) -> dict[str, int]:
    """Persist cleaned datasets into the ``processed`` schema.

    Cleaned tables mirror the raw layout, so they are (re)created directly
    from the frames rather than through the ORM.
    """
    engine = engine or get_engine()
    create_schemas(engine)
    counts: dict[str, int] = {}
    for name in RAW_TABLES:
        if name not in cleaned:
            continue
        frame = _normalize_for_db(name, cleaned[name])
        frame.write_database(
            table_name=f"{PROCESSED_SCHEMA}.{name}",
            connection=engine,
            if_table_exists="replace",
            engine="sqlalchemy",
        )
        counts[name] = frame.height
        logger.info("Wrote %d cleaned rows into %s.%s", frame.height, PROCESSED_SCHEMA, name)
    return counts


def save_features_to_db(features: pl.DataFrame) -> int:
    """Replace ``processed.order_features`` with an engineered feature frame."""
    init_database()
    with session_scope() as session:
        return crud.replace_order_features(session, features)


def main() -> None:
    """CLI entrypoint: load raw CSVs and optionally the cleaned tables."""
    parser = argparse.ArgumentParser(description="Load Olist CSVs into PostgreSQL")
    parser.add_argument("--data-dir", required=True, help="Directory of raw CSV files")
    parser.add_argument(
        "--with-processed",
        action="store_true",
        help="Also clean the datasets and populate the processed schema",
    )
    args = parser.parse_args()
    configure_logging()

    datasets = load_datasets(args.data_dir)
    counts = load_raw_to_db(args.data_dir, datasets=datasets)
    logger.info("Raw load complete: %s", counts)

    if args.with_processed:
        cleaned = clean_datasets(datasets)
        processed_counts = write_processed_tables(cleaned)
        logger.info("Processed load complete: %s", processed_counts)


if __name__ == "__main__":
    main()
