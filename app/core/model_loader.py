from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import mlflow
from loguru import logger
from mlflow.entities import Run
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient
from app.core.config import settings


@dataclass
class ModelMetadata:
    """Metadata bundle for the loaded model."""

    name: str
    version: str
    run_id: str
    stage: str | None
    metrics: dict[str, float]
    parameters: dict[str, str]
    registered_at: datetime | None


class ModelLoader:
    """Singleton: loads ML model once from MLflow and caches it."""

    _instance: ModelLoader | None = None
    _model: PyFuncModel | None = None
    _metadata: ModelMetadata | None = None

    def __new__(cls) -> ModelLoader:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._model is None:
            self._load()

    def _load(self) -> None:
        """Load model from MLflow registry or latest run."""
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        client = MlflowClient()

        logger.info("Attempting to load model from MLflow...")

        # ─────────────────────────────────────────────────────────────
        # 1) Try Model Registry (Production stage)
        # ─────────────────────────────────────────────────────────────
        try:
            model_uri = f"models:/{settings.MODEL_REGISTRY_NAME}/Production"
            self._model = mlflow.pyfunc.load_model(model_uri)

            # Fetch version details
            mv = client.get_latest_versions(
                settings.MODEL_REGISTRY_NAME, stages=["Production"]
            )[0]
            run = client.get_run(mv.run_id)

            self._metadata = self._build_metadata(
                name=settings.MODEL_REGISTRY_NAME,
                version=mv.version,
                run=run,
                stage="Production",
                registered_at=datetime.fromtimestamp(mv.creation_timestamp / 1000),
            )

            logger.success(
                f"✓ Loaded model from Registry: {self._metadata.name} "
                f"v{self._metadata.version} (run={self._metadata.run_id[:8]})"
            )
            return
        except Exception as e:
            logger.warning(f"Registry load failed: {e}. Falling back to latest run.")

        # ─────────────────────────────────────────────────────────────
        # 2) Fallback: latest successful run from experiment
        # ─────────────────────────────────────────────────────────────
        try:
            experiment = client.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME)
            if not experiment:
                raise RuntimeError(
                    f"Experiment '{settings.MLFLOW_EXPERIMENT_NAME}' not found"
                )

            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=1,
            )

            if not runs:
                raise RuntimeError("No finished runs found in experiment")

            run = runs[0]
            model_uri = f"runs:/{run.info.run_id}/model"
            self._model = mlflow.pyfunc.load_model(model_uri)

            self._metadata = self._build_metadata(
                name=settings.MLFLOW_EXPERIMENT_NAME,
                version=f"run-{run.info.run_id[:8]}",
                run=run,
                stage=None,
                registered_at=None,
            )

            logger.success(
                f"✓ Loaded model from latest run: {self._metadata.run_id[:8]}"
            )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError("Model loading failed") from e

    def _build_metadata(
        self,
        name: str,
        version: str,
        run: Run,
        stage: str | None,
        registered_at: datetime | None,
    ) -> ModelMetadata:
        """Extract metadata from MLflow Run."""
        metrics = {k: float(v) for k, v in run.data.metrics.items()}
        parameters = {k: str(v) for k, v in run.data.params.items()}

        return ModelMetadata(
            name=name,
            version=version,
            run_id=run.info.run_id,
            stage=stage,
            metrics=metrics,
            parameters=parameters,
            registered_at=registered_at,
        )

    @property
    def model(self) -> PyFuncModel:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        return self._model

    @property
    def metadata(self) -> ModelMetadata:
        if self._metadata is None:
            raise RuntimeError("Model metadata not available")
        return self._metadata

    def predict(self, features: Any) -> Any:
        """Thin wrapper for prediction."""
        return self.model.predict(features)

    def reload(self) -> None:
        """Force reload from MLflow (for runtime updates)."""
        logger.info("Reloading model...")
        self._model = None
        self._metadata = None
        self._load()


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton instance
# ─────────────────────────────────────────────────────────────────────────────

_loader: ModelLoader | None = None


def get_model_loader() -> ModelLoader:
    """FastAPI dependency: returns the singleton model loader."""
    global _loader
    if _loader is None:
        _loader = ModelLoader()
    return _loader
