"""
Cleaner for marketing_qualified_leads.
"""
import polars as pl

from .base import BaseCleaner


class LeadsCleaner(BaseCleaner):
    def __init__(self):
        super().__init__("marketing_qualified_leads")

    def clean(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lf
            .with_columns([
                pl.col("first_contact_date").str.to_datetime(strict=False),
            ])
            .filter(pl.col("mql_id").is_not_null())
            # Fill null origin with "unknown"
            .with_columns([
                pl.col("origin").fill_null("unknown")
            ])
            .unique(subset=["mql_id"], keep="first")
        )
