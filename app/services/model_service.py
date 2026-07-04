"""Model lifecycle service.

Resolution order at startup (and on ``reload``):

1. Registered MLflow model at ``models:/{name}/{stage}``.
2. Local joblib artifact at ``settings.model_local_path``.
3. Deterministic heuristic baseline (service stays available, gauge drops).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.exceptions import ModelNotLoadedError
from ml.feature_engineering import FEATURE_COLUMNS
from ml.predict import HeuristicBaselineModel
from utils.logging import get_logger
from utils.metrics import MODEL_LOADED

logger = get_logger(__name__)

SOURCE_REGISTRY = "mlflow-registry"
SOURCE_LOCAL = "local-artifact"
SOURCE_FALLBACK = "heuristic-fallback"


class ModelService:
    """Thread-safe holder of the currently served model and its metadata."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: object | None = None
        self.model_name: str = "unloaded"
        self.model_version: str = "0"
        self.stage: str = "None"
        self.source: str = "none"
        self.loaded_at: datetime = datetime.now(timezone.utc)

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        """Load the best available model following the resolution order."""
        settings = get_settings()
        with self._lock:
            if self._try_load_registry():
                pass
            elif self._try_load_local(Path(settings.model_local_path)):
                pass
            else:
                self._use_fallback()
            self.loaded_at = datetime.now(timezone.utc)
            MODEL_LOADED.set(1 if self.source == SOURCE_REGISTRY else 0)
            logger.info(
                "Model ready: name=%s version=%s source=%s",
                self.model_name,
                self.model_version,
                self.source,
            )

    def reload(self) -> None:
        """Re-run model resolution (e.g. after a new registry promotion)."""
        self.load()

    def _try_load_registry(self) -> bool:
        settings = get_settings()
        try:
            from ml.registry import load_registered_model

            model, info = load_registered_model(settings.model_name, settings.model_stage)
            self._model = model
            self.model_name = info.name
            self.model_version = info.version
            self.stage = info.stage
            self.source = SOURCE_REGISTRY
            return True
        except Exception as exc:  # registry down, model missing, network error
            logger.warning("MLflow registry model unavailable: %s", exc)
            return False

    def _try_load_local(self, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            import joblib

            self._model = joblib.load(path)
            settings = get_settings()
            self.model_name = settings.model_name
            self.model_version = "local"
            self.stage = "Local"
            self.source = SOURCE_LOCAL
            return True
        except Exception as exc:
            logger.warning("Local model artifact unusable (%s): %s", path, exc)
            return False

    def _use_fallback(self) -> None:
        self._model = HeuristicBaselineModel()
        self.model_name = HeuristicBaselineModel.name
        self.model_version = "baseline"
        self.stage = "Fallback"
        self.source = SOURCE_FALLBACK

    # --------------------------------------------------------------- predict
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return the positive-class (late) probability for each row."""
        if self._model is None:
            raise ModelNotLoadedError("No model is loaded")
        probabilities = self._model.predict_proba(features)[:, 1]
        return np.asarray(probabilities, dtype=float)

    @property
    def feature_count(self) -> int:
        return len(FEATURE_COLUMNS)


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    """Return the process-wide model service singleton."""
    return ModelService()
