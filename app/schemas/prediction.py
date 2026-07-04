from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from typing import Literal


class PredictionInput(BaseModel):

    order_id: str
    item_total: float = Field(ge=0)
    freight_total: float = Field(ge=0)
    estimated_days: float = Field(gt=0, le=90)
    purchase_hour: int = Field(ge=0, le=23)
    purchase_weekday: int = Field(ge=0, le=6)


class PredictionResponse(BaseModel):

    order_id: str
    late_delivery_probability: float
    risk_level: Literal["low", "medium", "high"]
    recommended_action: str
    model_version: str
    latency: float


class PredictionInput(BaseModel):
    order_id: str
    item_total: float = Field(ge=0)
    freight_total: float = Field(ge=0)
    estimated_days: float = Field(gt=0, le=90)
    purchase_hour: int = Field(ge=0, le=23)
    purchase_weekday: int = Field(ge=0, le=6)

def predict(payload: PredictionInput) -> dict:
    # Reference baseline: bounded, explainable risk score; replace with MLflow Staging model in production.
    score = min(0.98, max(0.01, 0.06 + 0.018 * payload.estimated_days + 0.0012 * payload.freight_total + 0.012 * (payload.purchase_hour >= 18)))
    level: Literal['low', 'medium', 'high'] = 'high' if score >= .55 else 'medium' if score >= .25 else 'low'
    action = {'low': 'monitor normally', 'medium': 'confirm carrier capacity', 'high': 'prioritize fulfillment intervention'}[level]
    REQUESTS.labels(level).inc()
    return {'order_id': payload.order_id, 'late_delivery_probability': round(score, 4), 'risk_level': level, 'recommended_action': action, 'model_version': 'group04-reference-baseline', 'latency': 0.0}




class PredictionInput(BaseModel):
    order_id: str
    item_total: float = Field(ge=0)
    freight_total: float = Field(ge=0)
    estimated_days: float = Field(gt=0, le=90)
    purchase_hour: int = Field(ge=0, le=23)
    purchase_weekday: int = Field(ge=0, le=6)

def predict(payload: PredictionInput) -> dict:
    # Reference baseline: bounded, explainable risk score; replace with MLflow Staging model in production.
    score = min(0.98, max(0.01, 0.06 + 0.018 * payload.estimated_days + 0.0012 * payload.freight_total + 0.012 * (payload.purchase_hour >= 18)))
    level: Literal['low', 'medium', 'high'] = 'high' if score >= .55 else 'medium' if score >= .25 else 'low'
    action = {'low': 'monitor normally', 'medium': 'confirm carrier capacity', 'high': 'prioritize fulfillment intervention'}[level]
    REQUESTS.labels(level).inc()
    return {'order_id': payload.order_id, 'late_delivery_probability': round(score, 4), 'risk_level': level, 'recommended_action': action, 'model_version': 'group04-reference-baseline', 'latency': 0.0}



#─────────────────────────────────────────────────────────────────────────────
# Request
# ─────────────────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """Minimum input for a single-order late-delivery prediction.
    The service will look up pre-computed features from the feature table
    using order_id. All fields here are available *before* actual delivery.
    """

    order_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Olist order UUID",
        examples=["e481f51cbdc54678b7cc49136f2d6af7"],
    )


class FeatureOverride(BaseModel):
    """
    Optional: caller can supply raw features directly (e.g., for testing
    or when the order is not yet in the feature table).
    All delivery-date-derived columns are excluded (leakage).
    """

    # Order
    num_items: int = Field(ge=1)
    total_price: float = Field(ge=0.0)
    total_freight: float = Field(ge=0.0)
    total_order_value: float = Field(ge=0.0)

    # Product
    avg_product_volume_cm3: float = Field(ge=0.0)
    max_product_volume_cm3: float = Field(ge=0.0)
    avg_product_weight_g: float = Field(ge=0.0)
    total_product_weight_g: float = Field(ge=0.0)

    # Price ratios
    avg_price_per_item: float = Field(ge=0.0)
    avg_freight_to_price_ratio: float = Field(ge=0.0)

    # Payment
    num_payment_installments: int = Field(ge=1)
    max_payment_installments: int = Field(ge=1)
    total_payment_value: float = Field(ge=0.0)

    # Review (may be absent for new orders)
    avg_review_score: float | None = Field(default=None, ge=1.0, le=5.0)
    num_reviews: int = Field(default=0, ge=0)

    # Location
    customer_state: str = Field(min_length=2, max_length=2)
    seller_state: str = Field(min_length=2, max_length=2)
    same_state_customer_seller: Literal[0, 1] = 0

    @field_validator("customer_state", "seller_state", mode="before")
    @classmethod
    def upper_state(cls, v: str) -> str:
        return v.strip().upper()


class PredictionRequestWithFeatures(PredictionRequest):
    """Full request when feature override is provided."""
    features: FeatureOverride | None = None


# ─────────────────────────────────────────────
# Response
# ─────────────────────────────────────────────────────────────────────────────

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


def _risk_level(prob: float) -> RiskLevel:
    if prob < 0.35:
        return "LOW"
    if prob < 0.65:
        return "MEDIUM"
    return "HIGH"


_RECOMMENDATIONS: dict[RiskLevel, str] = {
    "LOW": "Order is on track. No action needed.",
    "MEDIUM": "Moderate delay risk. Consider notifying the customer proactively.",
    "HIGH": "High delay risk. Escalate to logistics team immediately.",
}


class PredictionResponse(BaseModel):
    order_id: str
    delay_probability: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    recommendation: str
    model_name: str
    model_version: str
    predicted_at: datetime

    @classmethod
    def build(
        cls,
        order_id: str,
        probability: float,
        model_name: str,
        model_version: str,
    ) -> "PredictionResponse":
        level = _risk_level(probability)
        return cls(
            order_id=order_id,
            delay_probability=round(probability, 4),
            risk_level=level,
            recommendation=_RECOMMENDATIONS[level],
            model_name=model_name,
            model_version=model_version,
            predicted_at=datetime.utcnow(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Batch (bonus)
# ─────────────────────────────────────────────

class BatchPredictionRequest(BaseModel):
    orders: list[PredictionRequestWithFeatures] = Field(
        min_length=1,
        max_length=100,
        description="Up to 100 orders per batch call",
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int = int
    errors: list[dict[str, str]] = Field(default_factory=list)
