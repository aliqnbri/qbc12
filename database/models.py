"""SQLAlchemy ORM models for the raw and processed schemas."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from database.session import PROCESSED_SCHEMA, RAW_SCHEMA


class Base(DeclarativeBase):
    """Declarative base shared by every table."""


# --------------------------------------------------------------------------
# raw schema — one table per source CSV
# --------------------------------------------------------------------------


class RawOrder(Base):
    __tablename__ = "orders"
    __table_args__ = {"schema": RAW_SCHEMA}

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    order_status: Mapped[str] = mapped_column(String(32), nullable=False)
    order_purchase_timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    order_approved_at: Mapped[datetime | None] = mapped_column(DateTime)
    order_delivered_carrier_date: Mapped[datetime | None] = mapped_column(DateTime)
    order_delivered_customer_date: Mapped[datetime | None] = mapped_column(DateTime)
    order_estimated_delivery_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class RawOrderItem(Base):
    __tablename__ = "order_items"
    __table_args__ = {"schema": RAW_SCHEMA}

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_item_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    seller_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    shipping_limit_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    freight_value: Mapped[float] = mapped_column(Float, nullable=False)


class RawOrderPayment(Base):
    __tablename__ = "order_payments"
    __table_args__ = {"schema": RAW_SCHEMA}

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    payment_sequential: Mapped[int] = mapped_column(Integer, primary_key=True)
    payment_type: Mapped[str | None] = mapped_column(String(32))
    payment_installments: Mapped[int] = mapped_column(Integer, nullable=False)
    payment_value: Mapped[float] = mapped_column(Float, nullable=False)


class RawOrderReview(Base):
    __tablename__ = "order_reviews"
    __table_args__ = {"schema": RAW_SCHEMA}

    review_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    review_score: Mapped[int] = mapped_column(Integer, nullable=False)
    review_comment_title: Mapped[str | None] = mapped_column(Text)
    review_comment_message: Mapped[str | None] = mapped_column(Text)
    review_creation_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    review_answer_timestamp: Mapped[datetime | None] = mapped_column(DateTime)


class RawCustomer(Base):
    __tablename__ = "customers"
    __table_args__ = {"schema": RAW_SCHEMA}

    customer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_unique_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    customer_zip_code_prefix: Mapped[str | None] = mapped_column(String(8))
    customer_city: Mapped[str] = mapped_column(String(128), nullable=False)
    customer_state: Mapped[str] = mapped_column(String(4), nullable=False)


class RawSeller(Base):
    __tablename__ = "sellers"
    __table_args__ = {"schema": RAW_SCHEMA}

    seller_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    seller_zip_code_prefix: Mapped[str | None] = mapped_column(String(8))
    seller_city: Mapped[str] = mapped_column(String(128), nullable=False)
    seller_state: Mapped[str] = mapped_column(String(4), nullable=False)


class RawProduct(Base):
    __tablename__ = "products"
    __table_args__ = {"schema": RAW_SCHEMA}

    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_category_name: Mapped[str | None] = mapped_column(String(128))
    product_name_length: Mapped[int | None] = mapped_column(Integer)
    product_description_length: Mapped[int | None] = mapped_column(Integer)
    product_photos_qty: Mapped[int | None] = mapped_column(Integer)
    product_weight_g: Mapped[float | None] = mapped_column(Float)
    product_length_cm: Mapped[float | None] = mapped_column(Float)
    product_height_cm: Mapped[float | None] = mapped_column(Float)
    product_width_cm: Mapped[float | None] = mapped_column(Float)


class RawMarketingQualifiedLead(Base):
    __tablename__ = "marketing_qualified_leads"
    __table_args__ = {"schema": RAW_SCHEMA}

    mql_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    first_contact_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    landing_page_id: Mapped[str | None] = mapped_column(String(64))
    origin: Mapped[str | None] = mapped_column(String(64))


class RawClosedDeal(Base):
    __tablename__ = "closed_deals"
    __table_args__ = {"schema": RAW_SCHEMA}

    mql_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    seller_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    sdr_id: Mapped[str | None] = mapped_column(String(64))
    sr_id: Mapped[str | None] = mapped_column(String(64))
    won_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    business_segment: Mapped[str | None] = mapped_column(String(128))
    lead_type: Mapped[str | None] = mapped_column(String(64))
    lead_behaviour_profile: Mapped[str | None] = mapped_column(String(64))
    has_company: Mapped[bool | None] = mapped_column(Boolean)
    has_gtin: Mapped[bool | None] = mapped_column(Boolean)
    declared_product_catalog_size: Mapped[float | None] = mapped_column(Float)
    declared_monthly_revenue: Mapped[float | None] = mapped_column(Float)


class RawProductCategoryTranslation(Base):
    __tablename__ = "product_category_name_translation"
    __table_args__ = {"schema": RAW_SCHEMA}

    product_category_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    product_category_name_english: Mapped[str] = mapped_column(String(128), nullable=False)


# --------------------------------------------------------------------------
# processed schema — engineered features and served predictions
# --------------------------------------------------------------------------


class OrderFeature(Base):
    """One engineered feature row per delivered order."""

    __tablename__ = "order_features"
    __table_args__ = {"schema": PROCESSED_SCHEMA}

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_purchase_timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, index=True
    )
    seller_id: Mapped[str | None] = mapped_column(String(64), index=True)
    customer_unique_id: Mapped[str | None] = mapped_column(String(64))
    late_delivery: Mapped[int | None] = mapped_column(Integer)

    price_total: Mapped[float | None] = mapped_column(Float)
    freight_total: Mapped[float | None] = mapped_column(Float)
    freight_ratio: Mapped[float | None] = mapped_column(Float)
    item_count: Mapped[float | None] = mapped_column(Float)
    distinct_sellers: Mapped[float | None] = mapped_column(Float)
    payment_installments: Mapped[float | None] = mapped_column(Float)
    payment_value_total: Mapped[float | None] = mapped_column(Float)
    estimated_days: Mapped[float | None] = mapped_column(Float)
    shipping_limit_days: Mapped[float | None] = mapped_column(Float)
    purchase_hour: Mapped[float | None] = mapped_column(Float)
    purchase_weekday: Mapped[float | None] = mapped_column(Float)
    purchase_month: Mapped[float | None] = mapped_column(Float)
    is_weekend: Mapped[float | None] = mapped_column(Float)
    total_weight_g: Mapped[float | None] = mapped_column(Float)
    max_weight_g: Mapped[float | None] = mapped_column(Float)
    total_volume_cm3: Mapped[float | None] = mapped_column(Float)
    avg_photos_qty: Mapped[float | None] = mapped_column(Float)
    avg_description_length: Mapped[float | None] = mapped_column(Float)
    zip_distance: Mapped[float | None] = mapped_column(Float)
    same_state: Mapped[float | None] = mapped_column(Float)
    seller_order_count: Mapped[float | None] = mapped_column(Float)
    seller_late_rate: Mapped[float | None] = mapped_column(Float)
    seller_avg_delay_days: Mapped[float | None] = mapped_column(Float)
    customer_order_count: Mapped[float | None] = mapped_column(Float)
    customer_state: Mapped[str | None] = mapped_column(String(4))
    seller_state: Mapped[str | None] = mapped_column(String(4))
    payment_type: Mapped[str | None] = mapped_column(String(32))
    product_category: Mapped[str | None] = mapped_column(String(128))

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class PredictionLog(Base):
    """Audit log of every prediction served by the API."""

    __tablename__ = "predictions"
    __table_args__ = {"schema": PROCESSED_SCHEMA}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    late_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    latency_seconds: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )
