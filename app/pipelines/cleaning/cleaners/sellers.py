"""
Cleaner for sellers table.
"""
import polars as pl

from .base import BaseCleaner


class SellersCleaner(BaseCleaner):
    def __init__(self):
        super().__init__("sellers")

    def clean(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lf
            .filter(pl.col("seller_id").is_not_null())
            .unique(subset=["seller_id"], keep="first")
        )
