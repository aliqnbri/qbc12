"""Leakage-safe feature engineering for the late-delivery model.

All features are computable at purchase time. Outcome-derived signals
(reviews, actual delivery timestamps) are never used as features; they only
contribute to the target and to *historical* seller statistics computed
strictly on the training window and joined forward in time.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl

from utils.logging import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "late_delivery"
ID_COLUMN = "order_id"
TIMESTAMP_COLUMN = "order_purchase_timestamp"

NUMERIC_FEATURES: list[str] = [
    "price_total",
    "freight_total",
    "freight_ratio",
    "item_count",
    "distinct_sellers",
    "payment_installments",
    "payment_value_total",
    "estimated_days",
    "shipping_limit_days",
    "purchase_hour",
    "purchase_weekday",
    "purchase_month",
    "is_weekend",
    "total_weight_g",
    "max_weight_g",
    "total_volume_cm3",
    "avg_photos_qty",
    "avg_description_length",
    "zip_distance",
    "same_state",
    "seller_order_count",
    "seller_late_rate",
    "seller_avg_delay_days",
    "customer_order_count",
]

CATEGORICAL_FEATURES: list[str] = [
    "customer_state",
    "seller_state",
    "payment_type",
    "product_category",
]

FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

#: Columns that would leak the outcome. Tests assert none of them is a feature.
FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {
        "review_score",
        "review_comment_title",
        "review_comment_message",
        "review_creation_date",
        "review_answer_timestamp",
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
        "order_approved_at",
        "order_status",
        "delay_days",
        TARGET_COLUMN,
    }
)

#: Non-feature columns carried through the training frame for splitting,
#: history computation and persistence.
HELPER_COLUMNS: list[str] = [
    ID_COLUMN,
    TIMESTAMP_COLUMN,
    "seller_id",
    "customer_unique_id",
    "delay_days",
    TARGET_COLUMN,
]


def build_target(orders: pl.DataFrame) -> pl.DataFrame:
    """Return delivered orders with the ``late_delivery`` target attached.

    ``late_delivery`` is 1 when the order reached the customer after the
    estimated delivery date. ``delay_days`` (actual minus estimated, in days)
    is kept only for seller-history computation, never as a model feature.
    """
    delivered = orders.filter(
        (pl.col("order_status") == "delivered")
        & pl.col("order_delivered_customer_date").is_not_null()
        & pl.col("order_estimated_delivery_date").is_not_null()
        & pl.col(TIMESTAMP_COLUMN).is_not_null()
    )
    return delivered.with_columns(
        (
            pl.col("order_delivered_customer_date")
            > pl.col("order_estimated_delivery_date")
        )
        .cast(pl.Int8)
        .alias(TARGET_COLUMN),
        (
            (
                pl.col("order_delivered_customer_date")
                - pl.col("order_estimated_delivery_date")
            ).dt.total_hours()
            / 24.0
        ).alias("delay_days"),
    )


def _translate_categories(
    products: pl.DataFrame, translation: pl.DataFrame | None
) -> pl.DataFrame:
    """Attach English category names to products when a translation exists."""
    if translation is None:
        return products.with_columns(
            pl.col("product_category_name").alias("product_category")
        )
    return products.join(
        translation.select("product_category_name", "product_category_name_english"),
        on="product_category_name",
        how="left",
    ).with_columns(
        pl.coalesce(
            pl.col("product_category_name_english"), pl.col("product_category_name")
        ).alias("product_category")
    )


def _aggregate_items(
    order_items: pl.DataFrame,
    products: pl.DataFrame,
    sellers: pl.DataFrame,
    translation: pl.DataFrame | None,
) -> pl.DataFrame:
    """Aggregate order items to one row per order with basket-level features.

    The *primary* seller/product of an order is the one attached to its most
    expensive item; its attributes (state, zip, category) represent the order.
    """
    products = _translate_categories(products, translation)
    # The public Olist dump misspells this column ("lenght"); normalize it.
    if "product_description_lenght" in products.columns:
        products = products.rename(
            {"product_description_lenght": "product_description_length"}
        )
    if "product_name_lenght" in products.columns:
        products = products.rename({"product_name_lenght": "product_name_length"})
    enriched = (
        order_items.join(
            products.select(
                "product_id",
                "product_category",
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm",
                "product_photos_qty",
                "product_description_length",
            ),
            on="product_id",
            how="left",
        )
        .join(
            sellers.select("seller_id", "seller_state", "seller_zip_code_prefix"),
            on="seller_id",
            how="left",
        )
        .with_columns(
            (
                pl.col("product_length_cm")
                * pl.col("product_height_cm")
                * pl.col("product_width_cm")
            ).alias("product_volume_cm3")
        )
        .sort(["order_id", "price"], descending=[False, True])
    )
    return enriched.group_by("order_id").agg(
        pl.col("price").sum().alias("price_total"),
        pl.col("freight_value").sum().alias("freight_total"),
        pl.len().cast(pl.Float64).alias("item_count"),
        pl.col("seller_id").n_unique().cast(pl.Float64).alias("distinct_sellers"),
        pl.col("shipping_limit_date").min().alias("shipping_limit_date"),
        pl.col("product_weight_g").sum().alias("total_weight_g"),
        pl.col("product_weight_g").max().alias("max_weight_g"),
        pl.col("product_volume_cm3").sum().alias("total_volume_cm3"),
        pl.col("product_photos_qty").mean().alias("avg_photos_qty"),
        pl.col("product_description_length").mean().alias("avg_description_length"),
        pl.col("seller_id").first().alias("seller_id"),
        pl.col("seller_state").first().alias("seller_state"),
        pl.col("seller_zip_code_prefix").first().alias("seller_zip_code_prefix"),
        pl.col("product_category").first().alias("product_category"),
    )


def _aggregate_payments(order_payments: pl.DataFrame) -> pl.DataFrame:
    """Aggregate payments to one row per order; the primary payment type is
    the one carrying the largest value."""
    return (
        order_payments.sort(["order_id", "payment_value"], descending=[False, True])
        .group_by("order_id")
        .agg(
            pl.col("payment_type").first().alias("payment_type"),
            pl.col("payment_installments")
            .max()
            .cast(pl.Float64)
            .alias("payment_installments"),
            pl.col("payment_value").sum().alias("payment_value_total"),
        )
    )


def _zip_to_int(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8).str.strip_chars().cast(pl.Int64, strict=False)


def build_training_frame(datasets: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Build the order-level modelling frame from the raw datasets.

    Returns one row per delivered order containing all purchase-time features,
    helper columns (ids, timestamp, delay) and the target. Seller/customer
    history features are *not* attached here — they must be joined per split
    via :func:`attach_history` to stay leakage-free.
    """
    required = {"orders", "order_items", "order_payments", "customers", "products", "sellers"}
    missing = required - set(datasets)
    if missing:
        raise KeyError(f"Missing required datasets: {sorted(missing)}")

    orders = build_target(datasets["orders"])
    items = _aggregate_items(
        datasets["order_items"],
        datasets["products"],
        datasets["sellers"],
        datasets.get("product_category_name_translation"),
    )
    payments = _aggregate_payments(datasets["order_payments"])
    customers = datasets["customers"].select(
        "customer_id", "customer_unique_id", "customer_state", "customer_zip_code_prefix"
    )

    frame = (
        orders.join(customers, on="customer_id", how="left")
        .join(items, on="order_id", how="inner")
        .join(payments, on="order_id", how="left")
    )

    frame = frame.with_columns(
        (
            (
                pl.col("order_estimated_delivery_date") - pl.col(TIMESTAMP_COLUMN)
            ).dt.total_hours()
            / 24.0
        ).alias("estimated_days"),
        (
            (pl.col("shipping_limit_date") - pl.col(TIMESTAMP_COLUMN)).dt.total_hours()
            / 24.0
        ).alias("shipping_limit_days"),
        pl.col(TIMESTAMP_COLUMN).dt.hour().cast(pl.Float64).alias("purchase_hour"),
        pl.col(TIMESTAMP_COLUMN).dt.weekday().cast(pl.Float64).alias("purchase_weekday"),
        pl.col(TIMESTAMP_COLUMN).dt.month().cast(pl.Float64).alias("purchase_month"),
        (pl.col(TIMESTAMP_COLUMN).dt.weekday() >= 6).cast(pl.Float64).alias("is_weekend"),
        (
            pl.col("freight_total")
            / (pl.col("price_total") + pl.lit(1e-6))
        ).alias("freight_ratio"),
        (_zip_to_int("customer_zip_code_prefix") - _zip_to_int("seller_zip_code_prefix"))
        .abs()
        .cast(pl.Float64)
        .alias("zip_distance"),
        (pl.col("customer_state") == pl.col("seller_state"))
        .cast(pl.Float64)
        .alias("same_state"),
    )

    columns = [
        ID_COLUMN,
        TIMESTAMP_COLUMN,
        "seller_id",
        "customer_unique_id",
        "delay_days",
        TARGET_COLUMN,
        *[c for c in FEATURE_COLUMNS if c not in ("seller_order_count", "seller_late_rate", "seller_avg_delay_days", "customer_order_count")],
    ]
    result = frame.select(columns)
    logger.info(
        "Built training frame: rows=%d late_rate=%.4f",
        result.height,
        result[TARGET_COLUMN].mean() or 0.0,
    )
    return result


def compute_seller_history(history: pl.DataFrame) -> pl.DataFrame:
    """Aggregate historical seller performance from a *past* window only.

    Args:
        history: Training-window frame containing ``seller_id``,
            ``late_delivery`` and ``delay_days``.

    Returns:
        One row per seller with order count, late rate and mean delay.
    """
    return history.group_by("seller_id").agg(
        pl.len().cast(pl.Float64).alias("seller_order_count"),
        pl.col(TARGET_COLUMN).mean().cast(pl.Float64).alias("seller_late_rate"),
        pl.col("delay_days").mean().alias("seller_avg_delay_days"),
    )


def compute_customer_history(history: pl.DataFrame) -> pl.DataFrame:
    """Aggregate historical customer activity from a *past* window only."""
    return history.group_by("customer_unique_id").agg(
        pl.len().cast(pl.Float64).alias("customer_order_count")
    )


def attach_history(
    frame: pl.DataFrame,
    seller_history: pl.DataFrame,
    customer_history: pl.DataFrame,
) -> pl.DataFrame:
    """Left-join precomputed history features; unseen entities stay null and
    are handled by the imputer inside the sklearn pipeline."""
    return frame.join(seller_history, on="seller_id", how="left").join(
        customer_history, on="customer_unique_id", how="left"
    )


def temporal_train_test_split(
    frame: pl.DataFrame, test_size: float = 0.2
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split chronologically by purchase timestamp: the most recent
    ``test_size`` share of orders becomes the evaluation set."""
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    ordered = frame.sort(TIMESTAMP_COLUMN)
    cutoff = int(ordered.height * (1.0 - test_size))
    train = ordered.head(cutoff)
    test = ordered.tail(ordered.height - cutoff)
    logger.info("Temporal split: train=%d test=%d", train.height, test.height)
    return train, test


def build_full_feature_frame(
    datasets: dict[str, pl.DataFrame], test_size: float = 0.2
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build leakage-safe train/test frames with history features attached.

    History statistics are computed on the training window only and joined to
    both splits, mimicking what would be known when the test orders were placed.
    """
    base = build_training_frame(datasets)
    train, test = temporal_train_test_split(base, test_size=test_size)
    seller_history = compute_seller_history(train)
    customer_history = compute_customer_history(train)
    train = attach_history(train, seller_history, customer_history)
    test = attach_history(test, seller_history, customer_history)
    return train, test


def _days_between(later: datetime | None, earlier: datetime | None) -> float | None:
    if later is None or earlier is None:
        return None
    return (later - earlier).total_seconds() / 86_400.0


def build_inference_frame(records: list[dict[str, Any]]) -> pl.DataFrame:
    """Build a feature frame for serving from validated request payloads.

    Each record uses the field names of
    :class:`app.api.schemas.prediction.PredictionRequest`. Missing optional
    values become nulls and are imputed by the persisted sklearn pipeline.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        purchase: datetime = record["order_purchase_timestamp"]
        item_count = float(record.get("item_count") or 1)
        price_total = float(record["price"])
        freight_total = float(record["freight_value"])
        weight = record.get("product_weight_g")
        length = record.get("product_length_cm")
        height = record.get("product_height_cm")
        width = record.get("product_width_cm")
        volume = (
            float(length) * float(height) * float(width) * item_count
            if None not in (length, height, width)
            else None
        )
        customer_zip = record.get("customer_zip_code_prefix")
        seller_zip = record.get("seller_zip_code_prefix")
        try:
            zip_distance: float | None = (
                abs(int(customer_zip) - int(seller_zip))
                if customer_zip is not None and seller_zip is not None
                else None
            )
        except (TypeError, ValueError):
            zip_distance = None
        customer_state = record.get("customer_state")
        seller_state = record.get("seller_state")
        same_state = (
            float(customer_state == seller_state)
            if customer_state is not None and seller_state is not None
            else None
        )
        payment_value = record.get("payment_value")
        rows.append(
            {
                "price_total": price_total,
                "freight_total": freight_total,
                "freight_ratio": freight_total / (price_total + 1e-6),
                "item_count": item_count,
                "distinct_sellers": float(record.get("distinct_sellers") or 1),
                "payment_installments": (
                    float(record["payment_installments"])
                    if record.get("payment_installments") is not None
                    else None
                ),
                "payment_value_total": (
                    float(payment_value)
                    if payment_value is not None
                    else price_total + freight_total
                ),
                "estimated_days": _days_between(
                    record["order_estimated_delivery_date"], purchase
                ),
                "shipping_limit_days": _days_between(
                    record.get("shipping_limit_date"), purchase
                ),
                "purchase_hour": float(purchase.hour),
                "purchase_weekday": float(purchase.isoweekday()),
                "purchase_month": float(purchase.month),
                "is_weekend": float(purchase.isoweekday() >= 6),
                "total_weight_g": float(weight) * item_count if weight is not None else None,
                "max_weight_g": float(weight) if weight is not None else None,
                "total_volume_cm3": volume,
                "avg_photos_qty": (
                    float(record["product_photos_qty"])
                    if record.get("product_photos_qty") is not None
                    else None
                ),
                "avg_description_length": (
                    float(record["product_description_length"])
                    if record.get("product_description_length") is not None
                    else None
                ),
                "zip_distance": float(zip_distance) if zip_distance is not None else None,
                "same_state": same_state,
                "seller_order_count": (
                    float(record["seller_order_count"])
                    if record.get("seller_order_count") is not None
                    else None
                ),
                "seller_late_rate": (
                    float(record["seller_late_rate"])
                    if record.get("seller_late_rate") is not None
                    else None
                ),
                "seller_avg_delay_days": (
                    float(record["seller_avg_delay_days"])
                    if record.get("seller_avg_delay_days") is not None
                    else None
                ),
                "customer_order_count": (
                    float(record["customer_order_count"])
                    if record.get("customer_order_count") is not None
                    else None
                ),
                "customer_state": customer_state,
                "seller_state": seller_state,
                "payment_type": record.get("payment_type"),
                "product_category": record.get("product_category"),
            }
        )

    schema: dict[str, pl.DataType] = {name: pl.Float64() for name in NUMERIC_FEATURES}
    schema.update({name: pl.Utf8() for name in CATEGORICAL_FEATURES})
    return pl.DataFrame(rows, schema=schema).select(FEATURE_COLUMNS)
