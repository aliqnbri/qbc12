"""
Cleaner for order_payments table.
"""
import polars as pl

from .base import BaseCleaner


class PaymentsCleaner(BaseCleaner):
    def __init__(self):
        super().__init__("order_payments")

    def clean(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lf
            .filter(
                pl.col("order_id").is_not_null() &
                (pl.col("payment_value") >= 0)
            )
            # No duplicates per order + payment_sequential
            .unique(subset=["order_id", "payment_sequential"], keep="first")
        )
