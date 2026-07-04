from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.config import settings
from api.core.model_loader import ModelLoader, get_model_loader
from api.database import get_db
from api.schemas.health import ComponentStatus, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    db: AsyncSession = Depends(get_db),
    model_loader: ModelLoader = Depends(get_model_loader),
) -> HealthResponse:
    """Health check endpoint."""

    # Check database
    try:
        await db.execute(text("SELECT 1"))
        db_status = ComponentStatus(status="ok")
    except Exception as e:
        db_status = ComponentStatus(status="unavailable", detail=str(e))

    # Check model
    try:
        _ = model_loader.metadata
        model_status = ComponentStatus(status="ok")
    except Exception as e:
        model_status = ComponentStatus(status="unavailable", detail=str(e))

    # Overall status
    if db_status.status == "ok" and model_status.status == "ok":
        overall = "healthy"
    elif db_status.status == "unavailable" or model_status.status == "unavailable":
        overall = "unhealthy"
    else:
        overall = "degraded"

    return HealthResponse(
        status=overall,
        version=settings.API_VERSION,
        model=model_status,
        database=db_status,
        timestamp=settings.get_timestamp(),
    )
