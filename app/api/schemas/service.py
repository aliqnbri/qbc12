"""Service-level schemas: root, health, model info, metrics summary."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RootResponse(BaseModel):
    """Service identity returned by ``GET /``."""

    service: str
    version: str
    description: str
    docs_url: str
    health_url: str


class HealthResponse(BaseModel):
    """Health probe result."""

    status: str
    model_loaded: bool
    model_source: str
    timestamp: datetime


class ModelInfoResponse(BaseModel):
    """Metadata of the currently served model."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    stage: str
    source: str
    loaded_at: datetime
    feature_count: int
    temporal_leakage_policy: str


class MetricsSummaryResponse(BaseModel):
    """Human-readable summary of serving counters."""

    model_config = ConfigDict(protected_namespaces=())

    generated_at: datetime
    total_predictions: int
    risk_distribution: dict[str, int]
    total_errors: int
    model_name: str
    model_version: str
