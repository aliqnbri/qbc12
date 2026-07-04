# File: app/api/routes/prediction.py (updated)
"""
Prediction endpoint with Prometheus instrumentation.
"""
import time
from typing import List, Optional

import mlflow
import polars as pl
from fastapi import APIRouter, HTTPException
from loguru import logger

from app.api.models.prediction import PredictionRequest, PredictionResponse
from app.monitoring.metrics import (
    prediction_counter,
    prediction_latency,
    prediction_probability_histogram,
    model_version_gauge,
    model_load_timestamp,
    error_counter
)

router = APIRouter(prefix="/predict", tags=["prediction"])

# Global model cache
_model = None
_model_version = None
_model_name = "late_delivery_predictor"
_model_stage = "Production"


def load_model(model_name: str = None, stage: str = None):
    """Load model from MLflow Model Registry and update metrics."""
    global _model, _model_version, _model_name, _model_stage
    
    if model_name:
        _model_name = model_name
    if stage:
        _model_stage = stage
    
    try:
        model_uri = f"models:/{_model_name}/{_model_stage}"
        _model = mlflow.pyfunc.load_model(model_uri)
        
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(_model_name, stages=[_model_stage])
        _model_version = versions[0].version if versions else "unknown"
        
        # Update Prometheus metrics
        model_version_gauge.labels(
            model_name=_model_name,
            stage=_model_stage
        ).set(float(_model_version) if _model_version.isdigit() else 0)
        
        model_load_timestamp.set(time.time())
        
        logger.info(f"Loaded model {_model_name} v{_model_version} from {_model_stage}")
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        error_counter.labels(
            error_type="ModelLoadError",
            endpoint="/predict"
        ).inc()
        raise


@router.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@router.get("/model-info")
async def model_info():
    """
    Get current model information.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": _model_name,
        "model_version": _model_version,
        "stage": _model_stage,
        "status": "loaded"
    }


@router.post("/", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict late delivery probability for an order.
    """
    if _model is None:
        load_model()
    
    start_time = time.time()
    
    try:
        # Convert request to DataFrame
        input_df = pl.DataFrame([request.dict()])
        
        # Predict
        prediction = _model.predict(input_df)
        proba = float(prediction[0]) if hasattr(prediction, "__getitem__") else float(prediction)
        
        # Track probability distribution
        prediction_probability_histogram.observe(proba)
        
        # Determine risk level
        if proba < 0.3:
            risk_level = "low"
            recommendation = "Order is expected to arrive on time."
        elif proba < 0.7:
            risk_level = "medium"
            recommendation = "Monitor order closely for potential delays."
        else:
            risk_level = "high"
            recommendation = "Proactive communication with customer recommended."
        
        # Track prediction count by risk level
        prediction_counter.labels(risk_level=risk_level).inc()
        
        # Track latency
        latency = time.time() - start_time
        prediction_latency.observe(latency)
        
        return PredictionResponse(
            order_id=request.order_id,
            late_probability=proba,
            risk_level=risk_level,
            model_version=_model_version,
            recommendation=recommendation
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        error_counter.labels(
            error_type=type(e).__name__,
            endpoint="/predict"
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=List[PredictionResponse])
async def predict_batch(requests: List[PredictionRequest]):
    """
    Batch prediction endpoint (bonus feature).
    """
    if _model is None:
        load_model()
    
    from app.monitoring.metrics import batch_prediction_counter, batch_prediction_size
    
    batch_prediction_counter.inc()
    batch_prediction_size.observe(len(requests))
    
    results = []
    
    for req in requests:
        try:
            pred = await predict(req)
            results.append(pred)
        except Exception as e:
            logger.error(f"Batch prediction failed for order {req.order_id}: {e}")
            # Continue with remaining predictions
    
    return results
