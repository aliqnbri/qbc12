"""Prometheus exposition and human-readable metrics summary."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from starlette.responses import Response

from app.api.schemas.service import MetricsSummaryResponse
from app.services.model_service import ModelService, get_model_service
from utils.metrics import PREDICTION_ERRORS_TOTAL, PREDICTIONS_TOTAL

router = APIRouter(tags=["monitoring"])


def _counter_by_label(counter: Counter, label: str) -> dict[str, int]:
    """Snapshot a labelled counter as ``{label_value: count}``."""
    snapshot: dict[str, int] = {}
    for metric in counter.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total") and label in sample.labels:
                snapshot[sample.labels[label]] = int(sample.value)
    return snapshot


@router.get("/metrics")
def metrics() -> Response:
    """Prometheus exposition endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/metrics-summary", response_model=MetricsSummaryResponse)
def metrics_summary(
    model_service: ModelService = Depends(get_model_service),
) -> MetricsSummaryResponse:
    """Compact JSON summary of serving counters."""
    risk_distribution = _counter_by_label(PREDICTIONS_TOTAL, "risk_level")
    errors = _counter_by_label(PREDICTION_ERRORS_TOTAL, "error_type")
    return MetricsSummaryResponse(
        generated_at=datetime.now(timezone.utc),
        total_predictions=sum(risk_distribution.values()),
        risk_distribution=risk_distribution,
        total_errors=sum(errors.values()),
        model_name=model_service.model_name,
        model_version=model_service.model_version,
    )
