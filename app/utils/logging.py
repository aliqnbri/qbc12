"""Structured logging configuration."""
import logging
import sys
from typing import Any

from loguru import logger

from api.core.config import get_settings


class InterceptHandler(logging.Handler):
    """Intercept standard logging and route to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging() -> None:
    """Configure application logging."""
    settings = get_settings()

    # Remove default logger
    logger.remove()

    # Add custom handler
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    if settings.log_format == "json":
        logger.add(
            sys.stdout,
            format="{message}",
            level=settings.log_level,
            serialize=True,
        )
    else:
        logger.add(
            sys.stdout,
            format=log_format,
            level=settings.log_level,
            colorize=True,
        )

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"]:
        logging.getLogger(logger_name).handlers = [InterceptHandler()]
