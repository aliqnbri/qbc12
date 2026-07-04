"""SQLModel definitions for raw schema tables.

Column names here are canonical and match scripts.schema_registry exactly.
Any spelling mismatch between source CSV and these models is resolved upstream
in data_transformer.rename_csv_aliases, not here.
"""
from datetime import datetime, date
from typing import Optional

from sqlmodel import SQLModel, Field
from app.core.config import settings


class OrderRaw(SQLModel, table=True):
    __tablename__ = "orders"
    __table_args__ = {"schema": settings.postgres_schema_raw}

    order_id: str = Field(primary_key=True)
    customer_id: str
    order_status: Optional[str] = None
    order_purchase_timestamp: datetime
    order_approved_at: Optional[datetime] = None
    order_delivered_carrier_date: Optional[datetime] = None
    order_delivered_customer_date: Optional[datetime] = None
    order_estimated_delivery_date: datetime


class OrderItemRaw(SQLModel, table=True):
    __tablename__ = "order_items"
    __table_args__ = {"schema": settings.postgres_schema_raw}

    order_id: str = Field(primary_key=True)
    order_item_id: int = Field(primary_key=True)
    product_id: str
    seller_id: str
    shipping_limit_date: datetime
    price: float
    freight_value: float


class OrderReviewRaw(SQLModel, table=True):
    __tablename__ = "order_reviews"
    __table_args__ = {"schema": settings.postgres_schema_raw}

    review_id: str = Field(primary_key=True)
    order_id: str
    review_score: Optional[int] = None
    review_comment_title: Optional[str] = None
    review_comment_message: Optional[str] = None
    review_creation_date: datetime
    review_answer_timestamp: Optional[datetime] = None


class CustomerRaw(SQLModel, table=True):
    __tablename__ = "customers"
    __table_args__ = {"schema": settings.postgres_schema_raw}

    customer_id: str = Field(primary_key=True)
    customer_unique_id: str
    customer_zip_code_prefix: str
    customer_city: str
    customer_state: str


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
    seller_zip_code_prefix: str
    seller_city: str
    seller_state: str


class OrderPaymentRaw(SQLModel, table=True):
    __tablename__ = "order_payments"
    __table_args__ = {"schema": settings.postgres_schema_raw}

    order_id: str = Field(primary_key=True)
    payment_sequential: int = Field(primary_key=True)
    payment_type: str
    payment_installments: int
    payment_value: float


class MarketingQualifiedLeadRaw(SQLModel, table=True):
    __tablename__ = "marketing_qualified_leads"
    __table_args__ = {"schema": settings.postgres_schema_raw}

    mql_id: str = Field(primary_key=True)
    first_contact_date: date
    landing_page_id: str
    origin: Optional[str] = None


class ClosedDealRaw(SQLModel, table=True):
    __tablename__ = "closed_deals"
    __table_args__ = {"schema": settings.postgres_schema_raw}

    mql_id: str = Field(primary_key=True)
    seller_id: str
    sdr_id: Optional[str] = None
    sr_id: Optional[str] = None
    won_date: datetime
    business_segment: Optional[str] = None
    lead_type: Optional[str] = None
    lead_behaviour_profile: Optional[str] = None
    has_company: Optional[bool] = None
    has_gtin: Optional[bool] = None
    average_stock: Optional[str] = None
    business_type: Optional[str] = None
    declared_product_catalog_size: Optional[float] = None
    declared_monthly_revenue: Optional[float] = None


class ProductCategoryNameTranslationRaw(SQLModel, table=True):
    __tablename__ = "product_category_name_translation"
    __table_args__ = {"schema": settings.postgres_schema_raw}

    product_category_name: str = Field(primary_key=True)
    product_category_name_english: str
