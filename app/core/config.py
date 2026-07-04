"""Application configuration.

Every value is sourced from environment variables (or an ``.env`` file) via
pydantic-settings. Nothing is hardcoded; see ``.env.example`` for the full
list of variables.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),  # allow model_name / model_stage fields
    )

    # --- Service -----------------------------------------------------------
    app_name: str = Field(default="olist-late-delivery-api")
    app_version: str = Field(default="1.0.0")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # --- Data --------------------------------------------------------------
    data_dir: str = Field(default="data")
    artifacts_dir: str = Field(default="artifacts")

    # --- PostgreSQL --------------------------------------------------------
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="olist")
    postgres_password: str = Field(default="")
    postgres_db: str = Field(default="olist")
    enable_db_logging: bool = Field(
        default=False,
        description="Persist served predictions into processed.predictions.",
    )

    # --- MLflow ------------------------------------------------------------
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="olist-late-delivery")
    model_name: str = Field(default="olist-late-delivery")
    model_stage: str = Field(default="Staging")
    model_local_path: str = Field(
        default="artifacts/model.joblib",
        description="Fallback model path used when the MLflow registry is unreachable.",
    )

    # --- Training ----------------------------------------------------------
    random_seed: int = Field(default=42)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    decision_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """SQLAlchemy connection URL for the application database."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()
