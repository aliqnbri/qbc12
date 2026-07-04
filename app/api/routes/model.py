from fastapi import APIRouter, Depends

from api.core.model_loader import ModelLoader, get_model_loader
from api.schemas.model_info import ModelInfoResponse

router = APIRouter(tags=["model"])


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info(
    model_loader: ModelLoader = Depends(get_model_loader),
) -> ModelInfoResponse:
    """Return current model metadata."""
    meta = model_loader.metadata

    return ModelInfoResponse(
        model_name=meta.name,
        model_version=meta.version,
        run_id=meta.run_id,
        stage=meta.stage,
        feature_version="v1",  # from features.py
        metrics=meta.metrics,
        parameters=meta.parameters,
        registered_at=meta.registered_at,
    )
