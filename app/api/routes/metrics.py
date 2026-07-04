# File: app/api/routes/metrics.py
"""
Prometheus metrics endpoint.
"""
from fastapi import APIRouter, Response
from loguru import logger

from app.monitoring.metrics import get_metrics, get_content_type

router = APIRouter(tags=["monitoring"])


@router.get("/metrics")
async def metrics_endpoint():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format.
    """
    try:
        metrics = get_metrics()
        return Response(
            content=metrics,
            media_type=get_content_type()
        )
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return Response(
            content=f"# Error generating metrics: {e}",
            media_type="text/plain",
            status_code=500
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "service": "olist_late_delivery_predictor",
        "version": "1.0.0"
    }
