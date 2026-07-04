# app/models/processed.py
from sqlmodel import SQLModel, Field
from datetime import datetime
from app.core.config import settings


class CustomerProcessed(SQLModel, table=True):
    """
    Cleaned & joined customer data after ETL
    """
    __tablename__ = "customer_processed"
    __table_args__ = {"schema": settings.postgres_schema_processed}
    
    customer_unique_id: str = Field(primary_key=True)
    customer_zip_code_prefix: int
    customer_city: str
    customer_state: str
    no_of_orders: int
    purchased_approved: float | None = None
    delivered_estimated: float | None = None
    purchased_delivered: float | None = None
    no_of_products: int
    price: float
    freight_value: float
    product_weight_g: float | None = None
    product_length_cm: float | None = None
    product_height_cm: float | None = None
    product_width_cm: float | None = None
    geolocation_lat: float | None = None
    geolocation_lng: float | None = None
    payment_type: str | None = None
    payment_installments: int | None = None
    payment_value: float
    review_score: float | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FeatureStore(SQLModel, table=True):
    """
    Final features for ML model training & prediction (order-based)
    """
    __tablename__ = "feature_store"
    __table_args__ = {"schema": settings.postgres_schema_processed}
    
    order_id: str = Field(primary_key=True)
    days_to_estimated: float | None = None
    freight_ratio: float | None = None
    product_weight_g: float | None = None
    purchase_hour: int | None = None
    purchase_day_of_week: int | None = None
    target: bool | None = Field(default=None, index=True)
    feature_version: str = Field(default="v1")
    created_at: datetime = Field(default_factory=datetime.utcnow)
