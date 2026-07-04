"""
Cleaner for customers table.
"""
import polars as pl

from .base import BaseCleaner


class CustomersCleaner(BaseCleaner):
    """Clean customers table."""

    def __init__(self):
        super().__init__("customers")

    def clean(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Remove duplicates and nulls in critical fields."""
        return (
            lf
            .filter(
                pl.col("customer_id").is_not_null() &
                pl.col("customer_unique_id").is_not_null()
            )
            .unique(subset=["customer_id"], keep="first")
        )
