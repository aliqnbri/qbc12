"""Structured application exceptions and FastAPI exception handlers."""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from utils.metrics import PREDICTION_ERRORS_TOTAL


class AppError(Exception):
    """Base class for all application errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ModelNotLoadedError(AppError):
    """Raised when no model (registry or fallback) could be loaded."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_type = "model_not_loaded"


class PredictionError(AppError):
    """Raised when inference fails for a syntactically valid payload."""

    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_type = "prediction_error"


class DataValidationError(AppError):
    """Raised when input data fails semantic validation."""

    status_code = status.HTTP_400_BAD_REQUEST
    error_type = "data_validation_error"


def register_exception_handlers(app: FastAPI) -> None:
    """Attach JSON handlers for application errors to a FastAPI app."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        PREDICTION_ERRORS_TOTAL.labels(error_type=exc.error_type).inc()
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_type,
                "detail": exc.message,
                "path": str(request.url.path),
            },
        )
