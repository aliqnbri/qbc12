"""Prometheus metric definitions shared across the API."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS_TOTAL = Counter(
    "olist_http_requests_total",
    "Total HTTP requests handled by the API.",
    labelnames=("method", "endpoint", "status"),
)

REQUEST_LATENCY_SECONDS = Histogram(
    "olist_request_latency_seconds",
    "HTTP request latency in seconds.",
    labelnames=("method", "endpoint"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTIONS_TOTAL = Counter(
    "olist_predictions_total",
    "Total predictions served, labelled by risk level.",
    labelnames=("risk_level",),
)

PREDICTION_ERRORS_TOTAL = Counter(
    "olist_prediction_errors_total",
    "Total prediction failures.",
    labelnames=("error_type",),
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "olist_prediction_latency_seconds",
    "Model inference latency in seconds.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

PREDICTED_PROBABILITY = Histogram(
    "olist_predicted_probability",
    "Distribution of predicted late-delivery probabilities.",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

MODEL_LOADED = Gauge(
    "olist_model_loaded",
    "1 when a registered MLflow model is loaded, 0 when the fallback is in use.",
)
