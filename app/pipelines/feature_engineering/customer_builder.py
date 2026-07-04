"""
Customer-level feature engineering pipeline.
Produces aggregated features per customer_unique_id matching final.csv schema.
"""

from datetime import datetime
from typing import Dict
import polars as pl
from loguru import logger
from app.core.database import get_sync_conn, _pg_copy


class CustomerFeatureBuilder:
    """
    Builds customer-level aggregated features from raw tables.
    Output schema matches final.csv structure.
    """

    def __init__(self, feature_version: str = "customer_v1"):
        self.feature_version = feature_version

    def build_features(
        self, 
        source_schema: str = "processed",
        target_schema: str = "features",
        target_table: str = "customer_features",
        truncate: bool = True
    ) -> pl.DataFrame:
        """
        Main pipeline:
        1. Load processed tables
        2. Join
        3. Aggregate by customer
        4. Write to DB
        
        Returns:
            Customer-level DataFrame with columns matching final.csv
        """
        logger.info(f"🔧 Starting customer feature engineering (version: {self.feature_version})")
        
        # Step 1: Load processed tables
        logger.info("📥 Loading processed tables...")
        tables = self._load_tables(source_schema)
        
        # Step 2: Join all tables
        logger.info("🔗 Joining tables...")
        df_joined = self._join_tables(tables)
        
        # Step 3: Aggregate to customer level
        logger.info("📊 Aggregating by customer...")
        df_customer = self._aggregate_by_customer(df_joined)
        
        # Step 4: Add metadata
        logger.info("🏷️  Adding metadata...")
        df_customer = self._add_metadata(df_customer)
        
        # Step 5: Collect result
        logger.info("⚡ Collecting results...")
        df_result = df_customer.collect()
        
        logger.success(f"✅ Generated {len(df_result):,} customer records with {len(df_result.columns)} features")
        
        # Step 6: Write to DB
        self._write_to_db(df_result, target_schema, target_table, truncate)
        
        return df_result

    def _load_tables(self, schema: str) -> Dict[str, pl.LazyFrame]:
        """Load required tables from PostgreSQL."""
        with get_sync_conn() as conn:
            tables = {}
            table_names = [
                "orders",
                "order_items", 
                "customers",
                "products",
                "sellers",
                "order_payments",
                "order_reviews",
                "geolocation"
            ]
            
            for name in table_names:
                query = f'SELECT * FROM {schema}."{name}"'
                tables[name] = pl.read_database(query, conn).lazy()
                logger.debug(f"  ✓ Loaded {schema}.{name}")
        
        return tables

    def _join_tables(self, tables: Dict[str, pl.LazyFrame]) -> pl.LazyFrame:
        """
        Join all tables on appropriate keys.
        Anchor: orders → customers
        """
        # Pre-aggregate geolocation to avoid duplicates
        geo_agg = (
            tables["geolocation"]
            .group_by("geolocation_zip_code_prefix")
            .agg([
                pl.col("geolocation_lat").mean().alias("geolocation_lat"),
                pl.col("geolocation_lng").mean().alias("geolocation_lng")
            ])
        )
        
        df = (
            tables["orders"]
            .join(tables["customers"], on="customer_id", how="left")
            .join(tables["order_items"], on="order_id", how="left")
            .join(tables["products"], on="product_id", how="left")
            .join(tables["sellers"], on="seller_id", how="left")
            .join(tables["order_payments"], on="order_id", how="left")
            .join(tables["order_reviews"], on="order_id", how="left")
            .join(
                geo_agg,
                left_on="customer_zip_code_prefix",
                right_on="geolocation_zip_code_prefix",
                how="left"
            )
        )
        return df

    def _aggregate_by_customer(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Aggregate order-level data to customer-level.
        Output columns match final.csv schema.
        """
        df_agg = df.group_by("customer_unique_id").agg([
            # Customer info (first occurrence)
            pl.col("customer_zip_code_prefix").first().alias("customer_zip_code_prefix"),
            pl.col("customer_city").first().alias("customer_city"),
            pl.col("customer_state").first().alias("customer_state"),
            
            # Geolocation (mean)
            pl.col("geolocation_lat").mean().alias("geolocation_lat"),
            pl.col("geolocation_lng").mean().alias("geolocation_lng"),
            
            # Order counts
            pl.col("order_id").n_unique().alias("no_of_orders"),
            pl.col("product_id").count().alias("no_of_products"),
            
            # Time deltas (mean across all orders, in days)
            (
                (pl.col("order_approved_at") - pl.col("order_purchase_timestamp"))
                .dt.total_days()
                .mean()
                .alias("purchased_approved")
            ),
            (
                (pl.col("order_estimated_delivery_date") - pl.col("order_delivered_customer_date"))
                .dt.total_days()
                .mean()
                .alias("delivered_estimated")
            ),
            (
                (pl.col("order_delivered_customer_date") - pl.col("order_purchase_timestamp"))
                .dt.total_days()
                .mean()
                .alias("purchased_delivered")
            ),
            
            # Price aggregates (sum)
            pl.col("price").sum().alias("price"),
            pl.col("freight_value").sum().alias("freight_value"),
            
            # Product dimensions (mean)
            pl.col("product_weight_g").mean().alias("product_weight_g"),
            pl.col("product_length_cm").mean().alias("product_length_cm"),
            pl.col("product_height_cm").mean().alias("product_height_cm"),
            pl.col("product_width_cm").mean().alias("product_width_cm"),
            
            # Payment info (most common type, sum/mean for numeric)
            pl.col("payment_type").mode().first().alias("payment_type"),
            pl.col("payment_installments").mean().alias("payment_installments"),
            pl.col("payment_value").sum().alias("payment_value"),
            
            # Review score (mean)
            pl.col("review_score").mean().alias("review_score"),
        ])
        
        return df_agg

    def _add_metadata(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add pipeline metadata."""
        return df.with_columns([
            pl.lit(self.feature_version).alias("feature_version"),
            pl.lit(datetime.now()).alias("created_at")
        ])

    def _write_to_db(
        self, 
        df: pl.DataFrame, 
        schema: str,
        table: str,
        truncate: bool
    ):
        """Write result to processed schema using _pg_copy."""
        with get_sync_conn() as conn:
            rows = _pg_copy(
                df=df,
                schema=schema,
                table=table,
                conn=conn,
                truncate=truncate
            )
            logger.success(f"✅ Written {rows:,} customer records to {schema}.{table}")
