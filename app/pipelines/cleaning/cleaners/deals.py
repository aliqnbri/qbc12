"""
Cleaner for closed_deals.
"""
import polars as pl

from .base import BaseCleaner


class DealsCleaner(BaseCleaner):
    def __init__(self):
        super().__init__("closed_deals")

    def clean(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lf
            .filter(pl.col("mql_id").is_not_null())
            .with_columns([
                pl.col("won_date").cast(pl.Datetime, strict=False)
            ])
            .unique(subset=["mql_id"], keep="first")
        )