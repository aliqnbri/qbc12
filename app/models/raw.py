"""SQLModel definitions for raw and processed schemas.

Column *names* here are canonical — they match `scripts.schema_registry`
exactly. Any spelling mismatch between a source CSV and these models
(e.g. Olist's `product_name_lenght` typo) is resolved upstream in
`data_transformer.rename_csv_aliases`, not here. Keep it that way: the ORM
layer should only ever see clean, canonical column names.
"""
from datetime import datetime, timezone
from typing import Optional
from app.core.config import settings

from sqlmodel import SQLModel, Field

settings.postgres_schema_raw

PROCESSED_SCHEMA = "processed"


def _ts_now() -> datetime:
    """Current UTC timestamp (replaces deprecated `datetime.utcnow()`)."""
    return datetime.now(timezone.utc)


# ────────────────────────────────────────────────────────────────────── raw layer

class OrderRaw(SQLModel, table=True):
    __tablename__ = "orders"
    __table_args__ = {"schema": settings.postgres_schema_raw}

    order_id: str = Field(primary_key=True)
    customer_id: str = None
    order_status: Optional[str] = None
    order_purchase_timestamp: Optional[datetime] = None
    order_approved_at: Optional[datetime] = None
    order_delivered_carrier_date: Optional[datetime] = None
    order_delivered_customer_date: Optional[datetime] = None
    order_estimated_delivery_date: Optional[datetime] = None


class OrderItemRaw(SQLModel, table=True):
    __tablename__ = "order_items"
    __table_args__ = {"schema": settings.postgres_schema_raw}

    order_id: str = Field(primary_key=True)
    order_item_id: int = Field(primary_key=True)
    product_id: Optional[str] = None
    seller_id: Optional[str] = None
    shipping_limit_date: Optional[datetime] = None
    price: Optional[float] = None
    freight_value: Optional[float] = None


class OrderReviewRaw(SQLModel, table=True):
    __tablename__ = "order_reviews"
    __table_args__ = {"schema": settings.postgres_schema_raw}

    review_id: str = Field(primary_key=True)
    order_id: Optional[str] = None
    review_score: Optional[int] = None
    review_comment_title: Optional[str] = None
    review_comment_message: Optional[str] = None
    review_creation_date: Optional[datetime] = None
    review_answer_timestamp: Optional[datetime] = None


class CustomerRaw(SQLModel, table=True):
    __tablename__ = "customers"
    __table_args__ = {"schema": settings.postgres_schema_raw}

    customer_id: str = Field(primary_key=True)
    customer_unique_id: Optional[str] = None
    customer_zip_code_prefix: Optional[str] = None  # str preserves leading zeros
    customer_city: Optional[str] = None
    customer_state: Optional[str] = None


class MarketingLeadRaw(SQLModel, table=True):
    __tablename__ = "marketing_qualified_leads"
    __table_args__ = {"schema": settings.postgres_schema_raw}

    mql_id: str = Field(primary_key=True)
    first_contact_date: Optional[datetime] = None
    landing_page_id: Optional[str] = None
    origin: Optional[str] = None


class ProductRaw(SQLModel, table=True):
    __tablename__ = "products"
    __table_args__ = {"schema": settings.postgres_schema_raw}

    product_id: str = Field(primary_key=True)
    product_category_name: Optional[str] = None
    product_name_length: Optional[int] = None
    product_description_length: Optional[int] = None
    product_photos_qty: Optional[int] = None
    product_weight_g: Optional[int] = None
    product_length_cm: Optional[int] = None
    product_height_cm: Optional[int] = None
    product_width_cm: Optional[int] = None


class SellerRaw(SQLModel, table=True):
    __tablename__ = "sellers"
    __table_args__ = {"schema": settings.postgres_schema_raw}

    seller_id: str = Field(primary_key=True)
    seller_zip_code_prefix: Optional[str] = None
    seller_city: Optional[str] = None
    seller_state: Optional[str] = None


class OrderPaymentRaw(SQLModel, table=True):
    __tablename__ = "order_payments"
    __table_args__ = {"schema": settings.postgres_schema_raw}

    order_id: str = Field(primary_key=True)
    payment_sequential: int = Field(primary_key=True)
    payment_type: Optional[str] = None
    payment_installments: Optional[int] = None
    payment_value: Optional[float] = None


class ClosedDealRaw(SQLModel, table=True):
    __tablename__ = "closed_deals"
    __table_args__ = {"schema": settings.postgres_schema_raw}

    mql_id: str = Field(primary_key=True)
    seller_id: Optional[str] = None
    sdr_id: Optional[str] = None
    sr_id: Optional[str] = None
    won_date: Optional[datetime] = None
    business_segment: Optional[str] = None
    lead_type: Optional[str] = None
    lead_behaviour_profile: Optional[str] = None
    has_company: Optional[bool] = None
    has_gtin: Optional[bool] = None
    average_stock: Optional[str] = None  # source data is free-text/inconsistent
    business_type: Optional[str] = None
    declared_product_catalog_size: Optional[float] = None
    declared_monthly_revenue: Optional[float] = None


class CategoryTranslationRaw(SQLModel, table=True):
    __tablename__ = "product_category_name_translation"
    __table_args__ = {"schema": settings.postgres_schema_raw}

    product_category_name: str = Field(primary_key=True)
    product_category_name_english: Optional[str] = None
