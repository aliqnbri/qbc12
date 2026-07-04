# File: app/monitoring/middleware.py
"""
FastAPI middleware for automatic metrics collection.
"""
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from app.monitoring.metrics import (
    request_count,
    request_duration,
    active_requests,
    error_counter
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware to automatically track HTTP request metrics.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        method = request.method
        path = request.url.path
        
        # Skip metrics endpoint itself
        if path == "/metrics":
            return await call_next(request)
        
        # Track active requests
        active_requests.labels(method=method, endpoint=path).inc()
        
        start_time = time.time()
        status_code = 500
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        
        except Exception as e:
            logger.error(f"Request failed: {method} {path} - {e}")
            error_counter.labels(
                error_type=type(e).__name__,
                endpoint=path
            ).inc()
            raise
        
        finally:
            # Record metrics
            duration = time.time() - start_time
            
            request_count.labels(
                method=method,
                endpoint=path,
                status=status_code
            ).inc()
            
            request_duration.labels(
                method=method,
                endpoint=path
            ).observe(duration)
            
            active_requests.labels(method=method, endpoint=path).dec()
