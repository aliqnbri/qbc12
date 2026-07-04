# File: app/main.py
"""
FastAPI application with Prometheus monitoring.
"""
from fastapi import FastAPI
from loguru import logger

from app.api.routes import prediction, metrics ,ingestion
from app.monitoring.middleware import PrometheusMiddleware
from app.core.config import settings

# Create FastAPI app
app = FastAPI(
    title="Olist Late Delivery Predictor",
    description="MLOps API for predicting e-commerce delivery delays",
    version="1.0.0"
)

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware)

# Include routers
app.include_router(metrics.router)
app.include_router(prediction.router)
app.include_router(ingestion.router)


logger.info("FastAPI application started with Prometheus monitoring")
