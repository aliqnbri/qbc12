"""Prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.model_service import ModelService, get_model_service
from app.services.prediction_service import predict_orders

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    model_service: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    """Score a single order."""
    return predict_orders([payload], model_service)[0]


@router.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(
    payload: BatchPredictionRequest,
    model_service: ModelService = Depends(get_model_service),
) -> BatchPredictionResponse:
    """Score a batch of orders in one call."""
    predictions = predict_orders(payload.orders, model_service)
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))


@router.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    include_in_schema=False,
)
def batch_predict_alias(
    payload: BatchPredictionRequest,
    model_service: ModelService = Depends(get_model_service),
) -> BatchPredictionResponse:
    """Alias of ``/predict-batch`` kept for the course API contract."""
    return predict_batch(payload, model_service)
