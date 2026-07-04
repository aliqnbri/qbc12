from .health import router as health_router
from .model_info import router as model_info_router
from .predict import router as predict_router

__all__ = ["health_router", "model_info_router", "predict_router"]
