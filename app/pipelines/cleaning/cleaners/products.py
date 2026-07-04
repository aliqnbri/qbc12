"""
Cleaner for products table.
"""
import polars as pl

from .base import BaseCleaner


class ProductsCleaner(BaseCleaner):
    """Clean products with outlier handling for dimensions/weight."""

    def __init__(self):
        super().__init__("products")

    def clean(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        1. Fill nulls in dimensional fields with median
        2. Remove extreme outliers (e.g., weight > 99th percentile * 3)
        """
        return (
            lf
            .filter(pl.col("product_id").is_not_null())
            # Fill missing dimensions with 0 or median (simple strategy)
            .with_columns([
                pl.col("product_name_length").fill_null(0),
                pl.col("product_description_length").fill_null(0),
                pl.col("product_photos_qty").fill_null(0),
                pl.col("product_weight_g").fill_null(pl.col("product_weight_g").median()),
                pl.col("product_length_cm").fill_null(pl.col("product_length_cm").median()),
                pl.col("product_height_cm").fill_null(pl.col("product_height_cm").median()),
                pl.col("product_width_cm").fill_null(pl.col("product_width_cm").median()),
            ])
            # Remove unrealistic dimensions (simple rule: all > 0)
            .filter(
                (pl.col("product_weight_g") > 0) &
                (pl.col("product_length_cm") > 0) &
                (pl.col("product_height_cm") > 0) &
                (pl.col("product_width_cm") > 0)
            )
            .unique(subset=["product_id"], keep="first")
        )
