"""Central logging configuration for every entrypoint (API, Airflow, CLI)."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once with a consistent structured format.

    Args:
        level: Logging level name (``DEBUG``, ``INFO``, ``WARNING``...).
    """
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level.upper())
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, ensuring logging is configured."""
    configure_logging()
    return logging.getLogger(name)
