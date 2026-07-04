"""Dataset discovery and normalization utilities.

Loads every Olist CSV in a directory into a ``dict[str, pl.DataFrame]`` keyed
by the normalized dataset name (``olist_orders_dataset.csv`` -> ``orders``).
Datetime columns are parsed and zip-code prefixes are kept as strings so that
leading zeros survive.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# Columns that must be parsed as datetimes, per dataset.
_DATETIME_COLUMNS: dict[str, list[str]] = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "order_reviews": ["review_creation_date", "review_answer_timestamp"],
    "marketing_qualified_leads": ["first_contact_date"],
    "closed_deals": ["won_date"],
}

# Zip-code prefix columns that must stay Utf8 with leading zeros preserved.
_ZIP_COLUMNS: dict[str, str] = {
    "customers": "customer_zip_code_prefix",
    "sellers": "seller_zip_code_prefix",
}


def normalize_dataset_name(name: str) -> str:
    """Strip the ``olist_`` prefix and ``_dataset`` suffix from a file stem."""
    if name.startswith("olist_"):
        name = name[len("olist_") :]
    if name.endswith("_dataset"):
        name = name[: -len("_dataset")]
    return name


def _apply_schema(name: str, frame: pl.DataFrame) -> pl.DataFrame:
    """Cast datetime and zip-code columns of a raw dataset to canonical types."""
    exprs: list[pl.Expr] = []
    for column in _DATETIME_COLUMNS.get(name, []):
        if column in frame.columns and frame.schema[column] == pl.Utf8:
            exprs.append(
                pl.col(column).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
            )
    zip_column = _ZIP_COLUMNS.get(name)
    if zip_column and zip_column in frame.columns:
        exprs.append(pl.col(zip_column).cast(pl.Utf8).str.zfill(5))
    if exprs:
        frame = frame.with_columns(exprs)
    return frame


def load_datasets(data_dir: str | Path) -> dict[str, pl.DataFrame]:
    """Load every ``*.csv`` file in ``data_dir`` keyed by normalized name.

    Args:
        data_dir: Directory containing the Olist CSV files.

    Returns:
        Mapping of normalized dataset name to its Polars DataFrame.

    Raises:
        FileNotFoundError: If the directory does not exist or holds no CSVs.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    datasets: dict[str, pl.DataFrame] = {}
    for path in sorted(data_dir.glob("*.csv")):
        name = normalize_dataset_name(path.stem)
        frame = pl.read_csv(path, infer_schema_length=10_000)
        frame = _apply_schema(name, frame)
        datasets[name] = frame
        logger.info("Loaded dataset %-40s rows=%d cols=%d", name, frame.height, frame.width)

    if not datasets:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return datasets
