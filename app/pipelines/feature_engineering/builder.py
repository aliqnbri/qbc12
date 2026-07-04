# File: app/pipelines/feature_engineering/builder.py
"""
Feature engineering pipeline: processed tables → order-level feature table.
"""
from datetime import datetime
from typing import Dict

import polars as pl
from loguru import logger


class FeatureBuilder:
    """Build order-level features from processed tables."""
    
    def __init__(self, feature_version: str = "v1"):
        self.feature_version = feature_version
    
    def engineer_features(self, tables: Dict[str, pl.LazyFrame]) -> pl.LazyFrame:
        """
        Engineer order-level features from all tables.
        
        Args:
            tables: Dictionary of LazyFrames keyed by table name
        
        Returns:
            LazyFrame with one row per order_id and all engineered features
        """
        logger.info("Starting feature engineering pipeline")
        
        # Step 1: Join all tables with orders as anchor
        joined = self._join_tables(tables)
        
        # Step 2: Compute row-level metrics
        with_metrics = self._compute_delivery_metrics(joined)
        with_metrics = self._compute_product_metrics(with_metrics)
        with_metrics = self._compute_price_metrics(with_metrics)
        
        # Step 3: Create target
        with_target = self._create_target(with_metrics)
        
        # Step 4: Aggregate to order level
        aggregated = self._aggregate_by_order(with_target)
        
        # Step 5: Enrich with metadata
        final = self._add_metadata(aggregated)
        
        logger.info("Feature engineering complete")
        return final
    
    def _join_tables(self, tables: Dict[str, pl.LazyFrame]) -> pl.LazyFrame:
        """Join all tables with orders as anchor."""
        logger.info("Joining tables")
        
        orders = tables["orders"]
        
        # Join order_items
        joined = orders.join(tables["order_items"], on="order_id", how="left")
        
        # Join customers
        joined = joined.join(tables["customers"], on="customer_id", how="left")
        
        # Join products
        joined = joined.join(tables["products"], on="product_id", how="left")
        
        # Join sellers
        joined = joined.join(tables["sellers"], on="seller_id", how="left")
        
        # Join payments
        joined = joined.join(tables["order_payments"], on="order_id", how="left")
        
        # Join reviews
        joined = joined.join(tables["order_reviews"], on="order_id", how="left")
        
        return joined
    
    def _compute_delivery_metrics(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Compute delivery delay and processing time."""
        return df.with_columns([
            (
                pl.col("order_delivered_customer_date") - pl.col("order_estimated_delivery_date")
            ).dt.total_days().alias("delivery_delay_days"),
            (
                pl.col("order_delivered_customer_date") - pl.col("order_purchase_timestamp")
            ).dt.total_days().alias("processing_time_days"),
        ])
    
    def _compute_product_metrics(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Compute product volume and weight."""
        return df.with_columns([
            (
                pl.col("product_length_cm") * 
                pl.col("product_width_cm") * 
                pl.col("product_height_cm")
            ).alias("product_volume_cm3"),
        ])
    
    def _compute_price_metrics(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Compute price-related features."""
        return df.with_columns([
            (
                pl.col("price") / pl.col("order_id").count().over("order_id")
            ).alias("price_per_item"),
            pl.when(pl.col("price") > 0)
            .then(pl.col("freight_value") / pl.col("price"))
            .otherwise(1.0)
            .alias("freight_to_price_ratio"),
            (pl.col("price") + pl.col("freight_value")).alias("total_item_value"),
        ])
    
    def _create_target(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Create binary late delivery target."""
        return df.with_columns([
            (
                pl.col("order_delivered_customer_date") > pl.col("order_estimated_delivery_date")
            ).cast(pl.Int8).alias("is_late_delivery"),
        ])
    
    def _aggregate_by_order(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Aggregate to one row per order_id."""
        logger.info("Aggregating to order level")
        
        return df.group_by("order_id").agg([
            # Target
            pl.col("is_late_delivery").max().alias("target_late_delivery"),
            
            # Order metadata
            pl.col("order_status").first(),
            pl.col("order_purchase_timestamp").first(),
            pl.col("order_approved_at").first(),
            pl.col("order_delivered_carrier_date").first(),
            pl.col("order_delivered_customer_date").first(),
            pl.col("order_estimated_delivery_date").first(),
            
            # Delivery metrics
            pl.col("delivery_delay_days").mean().alias("avg_delivery_delay_days"),
            pl.col("delivery_delay_days").max().alias("max_delivery_delay_days"),
            pl.col("processing_time_days").mean().alias("avg_processing_time_days"),
            
            # Order value and items
            pl.col("total_item_value").sum().alias("total_order_value"),
            pl.col("price").sum().alias("total_price"),
            pl.col("freight_value").sum().alias("total_freight"),
            pl.col("order_item_id").count().alias("num_items"),
            
            # Product features
            pl.col("product_volume_cm3").mean().alias("avg_product_volume_cm3"),
            pl.col("product_volume_cm3").max().alias("max_product_volume_cm3"),
            pl.col("product_weight_g").mean().alias("avg_product_weight_g"),
            pl.col("product_weight_g").sum().alias("total_product_weight_g"),
            
            # Price features
            pl.col("price_per_item").mean().alias("avg_price_per_item"),
            pl.col("freight_to_price_ratio").mean().alias("avg_freight_to_price_ratio"),
            
            # Payment features
            pl.col("payment_sequential").max().alias("num_payment_installments"),
            pl.col("payment_installments").max().alias("max_payment_installments"),
            pl.col("payment_value").sum().alias("total_payment_value"),
            
            # Review features
            pl.col("review_score").mean().alias("avg_review_score"),
            pl.col("review_score").count().alias("num_reviews"),
            
            # Location features
            pl.col("customer_city").first(),
            pl.col("customer_state").first(),
            pl.col("seller_city").first(),
            pl.col("seller_state").first(),
            (
                pl.col("customer_state").first() == pl.col("seller_state").first()
            ).cast(pl.Int8).alias("same_state_customer_seller"),
        ])
    
    def _add_metadata(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add feature engineering metadata."""
        return df.with_columns([
            pl.lit(self.feature_version).alias("feature_version"),
            pl.lit(datetime.now()).alias("created_at"),
        ])
