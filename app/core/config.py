# core/config.py
"""
Configuration module for the MLOps pipeline.
Handles environment variables, database connections, and service settings.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal
from datetime import datetime

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with validation and environment variable support.
    Uses Pydantic v2 settings management for type safety and validation.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="QBC12 Group04 delivery-risk API", description="API title")
    app_version: str = Field(default="1.0.0", description="API version")
    environment: Literal["dev", "staging", "production"] = Field(default="dev")
    debug: bool = Field(default=False)

    # Database - PostgreSQL 16
    postgres_host: str = Field(default="postgres", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_user: str = Field(default="mlops_user")
    postgres_password: str = Field(default="mlops_password")
    postgres_db: str = Field(default="olist_mlops")
    postgres_pool_size: int = Field(default=10, ge=1, le=50)
    postgres_max_overflow: int = Field(default=20, ge=0, le=100)
    postgres_pool_pre_ping: bool = Field(default=True, description="Verify connections before use")
    postgres_echo: bool = Field(default=False, description="Log SQL statements")
    
    postgres_schema_raw: str = Field(default="raw", description="Raw data schema name")
    postgres_schema_processed: str = Field(default="processed", description="Processed features schema name")
    postgres_schema_predictions: str = Field(default="predictions", description="Predictions schema name")

    # Redis - caching layer
    redis_host: str = Field(default="redis")
    redis_port: int = Field(default=6379, ge=1, le=65535)
    redis_db: int = Field(default=0, ge=0)
    redis_password: str | None = Field(default=None)
    redis_cache_ttl: int = Field(default=3600, ge=0, description="Cache TTL in seconds")

    # Data paths - aligned with .env
    data_dir: Path = Field(default=Path("./data"), description="Root data directory")
    raw_data_dir: Path = Field(default=Path("./data/raw"), description="Raw CSV data location")
    processed_dir: Path = Field(default=Path("./data/processed"), description="Processed data output")    
    feature_store_path: Path = Field(default=Path("./data/feature_store"), description="Processed feature store")
    model_artifacts_path: Path = Field(default=Path("./models"), description="Model artifacts directory")

    # MLflow
    model_registry_name: str = Field(default="late_delivery")
    mlflow_tracking_uri: str = Field(default="http://mlflow:5000")
    mlflow_experiment_name: str = Field(default="delivery_risk")
    mlflow_artifact_location: str = Field(default="s3://mlflow-artifacts")

    
    # ONNX Runtime
    onnx_model_path: Path = Field(default=Path("./models/production.onnx"))
    onnx_providers: list[str] = Field(default=["CUDAExecutionProvider", "CPUExecutionProvider"])
    onnx_session_options_intra_threads: int = Field(default=4, ge=1)
    onnx_session_options_inter_threads: int = Field(default=4, ge=1)

    # Monitoring
    prometheus_multiproc_dir: Path | None = Field(default=None)
    metrics_port: int = Field(default=9090, ge=1, le=65535)

    # Airflow
    airflow_home: Path = Field(default=Path("./airflow"))
    airflow_dags_folder: Path = Field(default=Path("./dags"))
    
    @staticmethod
    def get_timestamp():
        return datetime.utcnow()

    @field_validator(
        "data_dir",
        "raw_data_dir",
        "processed_dir",
        "feature_store_path",
        "model_artifacts_path",
        "onnx_model_path",
        "airflow_home",
        "airflow_dags_folder",
        mode="after"
    )
    @classmethod
    def expand_path(cls, v: Path | None) -> Path | None:
        """Expand user home and resolve path."""
        if v is None:
            return None
        return v.expanduser().resolve()

    @property
    def database_url(self) -> str:
        """Construct synchronous database URL."""
        return str(
            PostgresDsn.build(
                scheme="postgresql",
                username=self.postgres_user,
                password=self.postgres_password,
                host=self.postgres_host,
                port=self.postgres_port,
                path=self.postgres_db,
            )
        )

    @property
    def async_database_url(self) -> str:
        """Construct async database URL for asyncpg."""
        return str(
            PostgresDsn.build(
                scheme="postgresql+asyncpg",
                username=self.postgres_user,
                password=self.postgres_password,
                host=self.postgres_host,
                port=self.postgres_port,
                path=self.postgres_db,
            )
        )

    @property
    def redis_url(self) -> str:
        """Construct Redis connection URL."""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance.
    Use this function to access settings throughout the application.
    Returns the same instance on every call (thread-safe).
    """
    return Settings()


# Export singleton
settings = get_settings()
