"""ETL cleaning of the raw Olist datasets.

Deduplicates primary keys, normalizes text, imputes structural gaps and caps
outliers with the IQR rule. The cleaned frames feed the ``processed`` schema
in PostgreSQL and the feature-engineering step.
"""

from __future__ import annotations

import polars as pl

from utils.logging import get_logger

logger = get_logger(__name__)

#: Numeric columns winsorized with the IQR rule, per dataset.
IQR_COLUMNS: dict[str, list[str]] = {
    "order_items": ["price", "freight_value"],
    "order_payments": ["payment_value"],
    "products": [
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ],
}

_PRIMARY_KEYS: dict[str, list[str]] = {
    "orders": ["order_id"],
    "order_items": ["order_id", "order_item_id"],
    "order_reviews": ["review_id"],
    "customers": ["customer_id"],
    "sellers": ["seller_id"],
    "products": ["product_id"],
    "marketing_qualified_leads": ["mql_id"],
    "product_category_name_translation": ["product_category_name"],
}

_CITY_COLUMNS: dict[str, str] = {
    "customers": "customer_city",
    "sellers": "seller_city",
}


def iqr_cap(frame: pl.DataFrame, column: str, factor: float = 1.5) -> pl.DataFrame:
    """Cap ``column`` to ``[q1 - factor*iqr, q3 + factor*iqr]`` (winsorize).

    Capping keeps every row (dropping outliers would bias the label
    distribution) while preventing extreme values from dominating scaling.
    Nulls are preserved for the downstream imputer.
    """
    if column not in frame.columns:
        return frame
    q1 = frame[column].quantile(0.25)
    q3 = frame[column].quantile(0.75)
    if q1 is None or q3 is None:
        return frame
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    capped = frame.with_columns(pl.col(column).clip(lower, upper).alias(column))
    n_capped = int(
        frame.select(((pl.col(column) < lower) | (pl.col(column) > upper)).sum()).item() or 0
    )
    if n_capped:
        logger.info("IQR-capped %d values in %s (bounds %.2f..%.2f)", n_capped, column, lower, upper)
    return capped


def _dedupe(name: str, frame: pl.DataFrame) -> pl.DataFrame:
    keys = _PRIMARY_KEYS.get(name)
    if not keys or any(key not in frame.columns for key in keys):
        return frame.unique(maintain_order=True)
    before = frame.height
    frame = frame.unique(subset=keys, keep="first", maintain_order=True)
    if frame.height != before:
        logger.info("Dropped %d duplicate rows from %s", before - frame.height, name)
    return frame


def _normalize_city(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    if column not in frame.columns:
        return frame
    return frame.with_columns(
        pl.col(column).cast(pl.Utf8).str.strip_chars().str.to_lowercase().alias(column)
    )


def clean_orders(orders: pl.DataFrame) -> pl.DataFrame:
    """Validate and clean the orders table.

    Rows with a missing purchase timestamp or an estimated delivery date that
    precedes the purchase are structurally invalid and dropped.
    """
    cleaned = _dedupe("orders", orders).filter(
        pl.col("order_purchase_timestamp").is_not_null()
        & pl.col("order_estimated_delivery_date").is_not_null()
        & (pl.col("order_estimated_delivery_date") >= pl.col("order_purchase_timestamp"))
    )
    logger.info("clean_orders: %d -> %d rows", orders.height, cleaned.height)
    return cleaned


def clean_order_items(order_items: pl.DataFrame) -> pl.DataFrame:
    """Clean order items: dedupe, drop non-positive prices, cap outliers."""
    cleaned = _dedupe("order_items", order_items).filter(
        (pl.col("price") > 0) & (pl.col("freight_value") >= 0)
    )
    for column in IQR_COLUMNS["order_items"]:
        cleaned = iqr_cap(cleaned, column)
    return cleaned


def clean_order_payments(order_payments: pl.DataFrame) -> pl.DataFrame:
    """Clean payments: dedupe, normalize unknown types, cap payment value."""
    cleaned = _dedupe("order_payments", order_payments).with_columns(
        pl.when(pl.col("payment_type").is_in(["not_defined", ""]))
        .then(pl.lit(None, dtype=pl.Utf8))
        .otherwise(pl.col("payment_type"))
        .alias("payment_type")
    )
    for column in IQR_COLUMNS["order_payments"]:
        cleaned = iqr_cap(cleaned, column)
    return cleaned


def clean_products(products: pl.DataFrame) -> pl.DataFrame:
    """Clean products: dedupe, cap physical attributes, median-impute weight
    and dimensions (nulls break volume computation downstream)."""
    cleaned = _dedupe("products", products)
    for column in IQR_COLUMNS["products"]:
        cleaned = iqr_cap(cleaned, column)
        if column in cleaned.columns:
            median = cleaned[column].median()
            cleaned = cleaned.with_columns(pl.col(column).fill_null(median).alias(column))
    return cleaned


def clean_customers(customers: pl.DataFrame) -> pl.DataFrame:
    """Clean customers: dedupe, normalize city, keep zip prefix as Utf8."""
    cleaned = _dedupe("customers", customers)
    cleaned = _normalize_city(cleaned, "customer_city")
    if "customer_zip_code_prefix" in cleaned.columns:
        cleaned = cleaned.with_columns(
            pl.col("customer_zip_code_prefix").cast(pl.Utf8).str.zfill(5)
        )
    return cleaned


def clean_sellers(sellers: pl.DataFrame) -> pl.DataFrame:
    """Clean sellers: dedupe, normalize city, keep zip prefix as Utf8."""
    cleaned = _dedupe("sellers", sellers)
    cleaned = _normalize_city(cleaned, "seller_city")
    if "seller_zip_code_prefix" in cleaned.columns:
        cleaned = cleaned.with_columns(
            pl.col("seller_zip_code_prefix").cast(pl.Utf8).str.zfill(5)
        )
    return cleaned


_CLEANERS = {
    "orders": clean_orders,
    "order_items": clean_order_items,
    "order_payments": clean_order_payments,
    "products": clean_products,
    "customers": clean_customers,
    "sellers": clean_sellers,
}


def clean_datasets(datasets: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Clean every raw dataset, applying dataset-specific rules where defined
    and plain deduplication otherwise."""
    cleaned: dict[str, pl.DataFrame] = {}
    for name, frame in datasets.items():
        cleaner = _CLEANERS.get(name)
        cleaned[name] = cleaner(frame) if cleaner else _dedupe(name, frame)
    return cleaned
