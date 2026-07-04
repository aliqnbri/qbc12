"""Service endpoints: root, health and model info."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.api.schemas.service import HealthResponse, ModelInfoResponse, RootResponse
from app.core.config import Settings, get_settings
from app.services.model_service import ModelService, get_model_service

router = APIRouter(tags=["service"])


@router.get("/", response_model=RootResponse)
def root(settings: Settings = Depends(get_settings)) -> RootResponse:
    """Service identity and useful links."""
    return RootResponse(
        service=settings.app_name,
        version=settings.app_version,
        description="Predicts the probability that an Olist order is delivered late.",
        docs_url="/docs",
        health_url="/health",
    )


@router.get("/health", response_model=HealthResponse)
def health(model_service: ModelService = Depends(get_model_service)) -> HealthResponse:
    """Liveness/readiness probe."""
    return HealthResponse(
        status="ok" if model_service.is_loaded else "degraded",
        model_loaded=model_service.is_loaded,
        model_source=model_service.source,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(
    model_service: ModelService = Depends(get_model_service),
) -> ModelInfoResponse:
    """Metadata of the currently served model."""
    return ModelInfoResponse(
        model_name=model_service.model_name,
        model_version=model_service.model_version,
        stage=model_service.stage,
        source=model_service.source,
        loaded_at=model_service.loaded_at,
        feature_count=model_service.feature_count,
        temporal_leakage_policy="purchase-time features only; outcome fields rejected",
    )
