# File: app/monitoring/metrics.py
"""
Prometheus metrics instrumentation for FastAPI service.
"""
from prometheus_client import Counter, Histogram, Gauge, Info
from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

# Create custom registry to avoid conflicts
registry = CollectorRegistry()

# Service info
service_info = Info(
    'fastapi_service',
    'FastAPI service information',
    registry=registry
)
service_info.info({
    'version': '1.0.0',
    'service': 'olist_late_delivery_predictor'
})

# Request metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    registry=registry
)

# Prediction metrics
prediction_counter = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['risk_level'],
    registry=registry
)

prediction_latency = Histogram(
    'prediction_duration_seconds',
    'Prediction latency in seconds',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    registry=registry
)

# Model metrics
model_version_gauge = Gauge(
    'model_version',
    'Current model version loaded',
    ['model_name', 'stage'],
    registry=registry
)

model_load_timestamp = Gauge(
    'model_load_timestamp_seconds',
    'Timestamp when model was last loaded',
    registry=registry
)

# Error metrics
error_counter = Counter(
    'errors_total',
    'Total number of errors',
    ['error_type', 'endpoint'],
    registry=registry
)

# Prediction distribution
prediction_probability_histogram = Histogram(
    'prediction_probability',
    'Distribution of predicted probabilities',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=registry
)

# Active requests
active_requests = Gauge(
    'http_requests_in_progress',
    'Number of HTTP requests in progress',
    ['method', 'endpoint'],
    registry=registry
)

# Batch prediction metrics (bonus)
batch_prediction_counter = Counter(
    'batch_predictions_total',
    'Total number of batch predictions',
    registry=registry
)

batch_prediction_size = Histogram(
    'batch_prediction_size',
    'Size of batch predictions',
    buckets=[1, 5, 10, 50, 100, 500, 1000],
    registry=registry
)


def get_metrics():
    """Generate Prometheus metrics in text format."""
    return generate_latest(registry)


def get_content_type():
    """Return Prometheus content type."""
    return CONTENT_TYPE_LATEST
