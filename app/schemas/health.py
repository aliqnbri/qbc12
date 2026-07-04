from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ComponentStatus(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    model: ComponentStatus
    database: ComponentStatus
    timestamp: datetime

    @classmethod
    def healthy(cls, version: str) -> "HealthResponse":
        return cls(
            status="healthy",
            version=version,
            model=ComponentStatus(status="ok"),
            database=ComponentStatus(status="ok"),
            timestamp=datetime.utcnow(),
        )
