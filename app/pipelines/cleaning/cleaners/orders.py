"""
Cleaner for orders table.
Priority: temporal consistency + status validation for late-delivery target.
"""
import polars as pl

from .base import BaseCleaner


class OrdersCleaner(BaseCleaner):
    """Clean orders table with focus on delivery date logic."""

    def __init__(self):
        super().__init__("orders")

    def clean(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Cleaning steps:
        1. Parse all timestamp columns → explicit format + time_unit='us'
        2. Remove rows with missing critical fields
        3. Fix temporal inconsistencies
        4. Filter only delivered orders for training
        """
        # Olist timestamp format
        ts_format = "%Y-%m-%d %H:%M:%S"
        
        timestamp_cols = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]

        return (
            lf
            # Parse timestamps with explicit format so Polars writes TIMESTAMP to PG
            .with_columns([
                pl.col(c).str.to_datetime(
                    format=ts_format,
                    strict=False,
                    time_unit="us",  # microseconds → maps to PostgreSQL TIMESTAMP
                ).alias(c)
                for c in timestamp_cols
            ])
            # Remove rows where critical dates are null
            .filter(
                pl.col("order_purchase_timestamp").is_not_null() &
                pl.col("order_estimated_delivery_date").is_not_null()
            )
            # Keep only delivered orders (we need actual delivery date for target)
            .filter(
                (pl.col("order_status") == "delivered") &
                pl.col("order_delivered_customer_date").is_not_null()
            )
            # Business logic: delivered date must be >= purchase date
            .filter(
                pl.col("order_delivered_customer_date") >= pl.col("order_purchase_timestamp")
            )
            # Business logic: estimated delivery must be >= purchase date
            .filter(
                pl.col("order_estimated_delivery_date") >= pl.col("order_purchase_timestamp")
            )
            # Remove duplicates
            .unique(subset=["order_id"], keep="first")
        )
