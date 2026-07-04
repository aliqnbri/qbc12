"""Request/response schemas for prediction endpoints (Pydantic v2)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RiskLevel = Literal["low", "medium", "high"]


class PredictionRequest(BaseModel):
    """Purchase-time information about one order.

    Only fields available at purchase time are accepted — outcome fields
    (reviews, actual delivery dates) are deliberately absent from the
    contract, which enforces the no-leakage policy at the API boundary.
    """

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(min_length=1, max_length=64)
    order_purchase_timestamp: datetime
    order_estimated_delivery_date: datetime
    shipping_limit_date: datetime | None = None

    customer_state: str | None = Field(default=None, max_length=4)
    customer_zip_code_prefix: str | None = Field(default=None, max_length=8)
    seller_state: str | None = Field(default=None, max_length=4)
    seller_zip_code_prefix: str | None = Field(default=None, max_length=8)

    price: float = Field(gt=0, description="Total item price of the order")
    freight_value: float = Field(ge=0, description="Total freight of the order")
    item_count: int = Field(default=1, ge=1, le=100)
    distinct_sellers: int = Field(default=1, ge=1, le=100)

    payment_type: str | None = Field(default=None, max_length=32)
    payment_installments: int | None = Field(default=None, ge=0, le=48)
    payment_value: float | None = Field(default=None, ge=0)

    product_category: str | None = Field(default=None, max_length=128)
    product_weight_g: float | None = Field(default=None, ge=0)
    product_length_cm: float | None = Field(default=None, ge=0)
    product_height_cm: float | None = Field(default=None, ge=0)
    product_width_cm: float | None = Field(default=None, ge=0)
    product_photos_qty: float | None = Field(default=None, ge=0)
    product_description_length: float | None = Field(default=None, ge=0)

    seller_order_count: float | None = Field(default=None, ge=0)
    seller_late_rate: float | None = Field(default=None, ge=0, le=1)
    seller_avg_delay_days: float | None = None
    customer_order_count: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def check_temporal_consistency(self) -> "PredictionRequest":
        """The estimated delivery date must not precede the purchase."""
        if self.order_estimated_delivery_date < self.order_purchase_timestamp:
            raise ValueError(
                "order_estimated_delivery_date must be >= order_purchase_timestamp"
            )
        return self


class PredictionResponse(BaseModel):
    """One served prediction."""

    model_config = ConfigDict(protected_namespaces=())

    order_id: str
    late_probability: float = Field(ge=0, le=1)
    late_delivery_probability: float = Field(
        ge=0, le=1, description="Alias of late_probability (course API contract)."
    )
    risk_level: RiskLevel
    model_name: str
    model_version: str
    recommended_action: str
    latency_seconds: float = Field(ge=0)


class BatchPredictionRequest(BaseModel):
    """Batch of orders to score in one call."""

    model_config = ConfigDict(extra="forbid")

    orders: list[PredictionRequest] = Field(min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Batch scoring result."""

    predictions: list[PredictionResponse]
    count: int
