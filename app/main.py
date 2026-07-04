"""FastAPI application factory and entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request, Response

from app.api.routers import metrics as metrics_router
from app.api.routers import predict as predict_router
from app.api.routers import service as service_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.services.model_service import get_model_service
from utils.logging import configure_logging, get_logger
from utils.metrics import HTTP_REQUESTS_TOTAL, REQUEST_LATENCY_SECONDS

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging and load the model before serving traffic."""
    settings = get_settings()
    configure_logging(settings.log_level)
    get_model_service().load()
    logger.info("%s v%s ready", settings.app_name, settings.app_version)
    yield


def create_app() -> FastAPI:
    """Build the FastAPI application with routers, middleware and handlers."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Predicts late-delivery risk for Olist orders using purchase-time "
            "features only. Backed by an MLflow-registered sklearn pipeline."
        ),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def record_http_metrics(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        endpoint = getattr(route, "path", request.url.path)
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method, endpoint=endpoint, status=str(response.status_code)
        ).inc()
        REQUEST_LATENCY_SECONDS.labels(method=request.method, endpoint=endpoint).observe(
            perf_counter() - started
        )
        return response

    register_exception_handlers(app)
    app.include_router(service_router.router)
    app.include_router(predict_router.router)
    app.include_router(metrics_router.router)
    return app


app = create_app()
