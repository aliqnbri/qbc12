from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.model_loader import ModelLoader, get_model_loader
from api.database import get_db
from api.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequestWithFeatures,
    PredictionResponse,
)
from api.services.prediction_service import (
    BatchPredictionService,
    PredictionService,
)

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_order(
    request: PredictionRequestWithFeatures,
    db: AsyncSession = Depends(get_db),
    model_loader: ModelLoader = Depends(get_model_loader),
) -> PredictionResponse:
    """Predict late delivery for a single order."""

    service = PredictionService(db, model_loader)

    try:
        return await service.predict_single(request.order_id, request.features)
    except ValueError as e:
        logger.warning(f"Prediction failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    db: AsyncSession = Depends(get_db),
    model_loader: ModelLoader = Depends(get_model_loader),
) -> BatchPredictionResponse:
    """Predict late delivery for multiple orders (bonus feature)."""

    service = BatchPredictionService(db, model_loader)

    orders = [(o.order_id, o.features) for o in request.orders]
    predictions, errors = await service.predict_batch(orders)

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        failed=len(errors),
        errors=errors,
    )
