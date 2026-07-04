"""MLflow model registry helpers: register, promote and load models."""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
from mlflow.tracking import MlflowClient

from app.core.config import get_settings
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RegisteredModelInfo:
    """Metadata of a loaded registered model."""

    name: str
    version: str
    stage: str
    run_id: str
    source: str


def register_best_model(
    run_id: str, model_name: str | None = None, stage: str | None = None
) -> RegisteredModelInfo:
    """Register the ``model`` artifact of a run and promote it to a stage.

    Args:
        run_id: MLflow run holding the winning model artifact.
        model_name: Registry name (defaults to ``settings.model_name``).
        stage: Target stage (defaults to ``settings.model_stage``).

    Returns:
        Metadata of the newly registered version.
    """
    settings = get_settings()
    model_name = model_name or settings.model_name
    stage = stage or settings.model_stage
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    model_uri = f"runs:/{run_id}/model"
    version = mlflow.register_model(model_uri=model_uri, name=model_name)
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version.version,
        stage=stage,
        archive_existing_versions=True,
    )
    logger.info(
        "Registered %s version %s from run %s and promoted to %s",
        model_name,
        version.version,
        run_id,
        stage,
    )
    return RegisteredModelInfo(
        name=model_name,
        version=str(version.version),
        stage=stage,
        run_id=run_id,
        source=model_uri,
    )


def load_registered_model(
    model_name: str | None = None, stage: str | None = None
) -> tuple[object, RegisteredModelInfo]:
    """Load the latest model version of a registry stage.

    Returns:
        Tuple of the loaded sklearn pipeline and its registry metadata.

    Raises:
        RuntimeError: When no version exists in the requested stage.
    """
    settings = get_settings()
    model_name = model_name or settings.model_name
    stage = stage or settings.model_stage
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        raise RuntimeError(f"No model named {model_name!r} in stage {stage!r}")
    version = versions[0]
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
    info = RegisteredModelInfo(
        name=model_name,
        version=str(version.version),
        stage=stage,
        run_id=version.run_id or "",
        source=version.source or "",
    )
    logger.info("Loaded registered model %s v%s (%s)", info.name, info.version, stage)
    return model, info
