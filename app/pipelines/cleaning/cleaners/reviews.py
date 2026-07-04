"""
Cleaner for order_reviews table.
"""
import polars as pl

from .base import BaseCleaner


class ReviewsCleaner(BaseCleaner):
    def __init__(self):
        super().__init__("order_reviews")

    def clean(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return (
            lf
            .with_columns([
                pl.col("review_creation_date").str.to_datetime(strict=False),
                pl.col("review_answer_timestamp").str.to_datetime(strict=False),
            ])
            .filter(
                pl.col("order_id").is_not_null() &
                pl.col("review_score").is_not_null()
            )
            # Score must be 1-5
            .filter(pl.col("review_score").is_between(1, 5))
            .unique(subset=["review_id"], keep="first")
        )
