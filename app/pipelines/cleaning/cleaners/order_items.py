"""
Cleaner for order_items table.
"""
import polars as pl

from .base import BaseCleaner


class OrderItemsCleaner(BaseCleaner):
    """Clean order_items with focus on price/freight validation."""

    def __init__(self):
        super().__init__("order_items")

    def clean(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Cleaning:
        1. Parse shipping_limit_date
        2. Remove negative prices/freight
        3. Remove rows with missing required fields
        """
        return (
            lf
            .with_columns([
                pl.col("shipping_limit_date").str.to_datetime(strict=False).alias("shipping_limit_date"),
            ])
            .filter(
                pl.col("order_id").is_not_null() &
                pl.col("product_id").is_not_null() &
                pl.col("seller_id").is_not_null()
            )
            # Prices and freight must be >= 0
            .filter(
                (pl.col("price") >= 0) &
                (pl.col("freight_value") >= 0)
            )
            # Remove duplicates
            .unique(subset=["order_id", "order_item_id"], keep="first")
        )
