"""Prediction orchestration: features -> model -> risk policy -> response."""

from __future__ import annotations

from time import perf_counter

from app.api.schemas.prediction import PredictionRequest, PredictionResponse
from app.core.config import get_settings
from app.core.exceptions import PredictionError
from app.services.model_service import ModelService
from ml.feature_engineering import build_inference_frame
from ml.predict import assign_risk_level, recommended_action
from ml.preprocessing import to_model_input
from utils.logging import get_logger
from utils.metrics import (
    PREDICTED_PROBABILITY,
    PREDICTION_ERRORS_TOTAL,
    PREDICTION_LATENCY_SECONDS,
    PREDICTIONS_TOTAL,
)

logger = get_logger(__name__)


def predict_orders(
    requests: list[PredictionRequest], model_service: ModelService
) -> list[PredictionResponse]:
    """Score a list of validated order payloads.

    Raises:
        PredictionError: When feature building or inference fails.
    """
    started = perf_counter()
    try:
        frame = build_inference_frame([request.model_dump() for request in requests])
        probabilities = model_service.predict_proba(to_model_input(frame))
    except PredictionError:
        raise
    except Exception as exc:
        PREDICTION_ERRORS_TOTAL.labels(error_type=type(exc).__name__).inc()
        logger.exception("Inference failed for %d order(s)", len(requests))
        raise PredictionError(f"Inference failed: {exc}") from exc

    elapsed = perf_counter() - started
    PREDICTION_LATENCY_SECONDS.observe(elapsed)
    per_order_latency = elapsed / max(len(requests), 1)

    responses: list[PredictionResponse] = []
    for request, probability in zip(requests, probabilities):
        probability = float(min(max(probability, 0.0), 1.0))
        risk = assign_risk_level(probability)
        PREDICTIONS_TOTAL.labels(risk_level=risk).inc()
        PREDICTED_PROBABILITY.observe(probability)
        responses.append(
            PredictionResponse(
                order_id=request.order_id,
                late_probability=round(probability, 6),
                late_delivery_probability=round(probability, 6),
                risk_level=risk,
                model_name=model_service.model_name,
                model_version=model_service.model_version,
                recommended_action=recommended_action(risk),
                latency_seconds=round(per_order_latency, 6),
            )
        )

    _maybe_log_to_db(responses)
    return responses


def _maybe_log_to_db(responses: list[PredictionResponse]) -> None:
    """Persist predictions when DB logging is enabled; never fail a request."""
    settings = get_settings()
    if not settings.enable_db_logging:
        return
    try:
        from database import crud
        from database.session import session_scope

        with session_scope() as session:
            for response in responses:
                crud.log_prediction(
                    session,
                    order_id=response.order_id,
                    late_probability=response.late_probability,
                    risk_level=response.risk_level,
                    model_name=response.model_name,
                    model_version=response.model_version,
                    latency_seconds=response.latency_seconds,
                )
    except Exception as exc:
        logger.warning("Prediction DB logging skipped: %s", exc)
