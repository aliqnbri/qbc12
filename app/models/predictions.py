"""SQLModel definitions for predictions schema (model inference logging)."""
from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field
from app.core.config import settings


class PredictionLog(SQLModel, table=True):
    """
    Log of all predictions made by the API for monitoring and audit.
    """
    __tablename__ = "prediction_logs"
    __table_args__ = {"schema": "predictions"}

    # Auto-increment primary key
    id: Optional[int] = Field(default=None, primary_key=True)

    # Request info
    order_id: str = Field(index=True)
    request_timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Prediction result
    late_delivery_probability: float
    risk_level: str = Field(
        ...,
        description="Risk level: low, medium, or high",    
    )
    # Performance
    model_version: str
    latency_ms: float

