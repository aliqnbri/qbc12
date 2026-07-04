from datetime import datetime

from pydantic import BaseModel


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    run_id: str
    stage: str | None = None          # "Production", "Staging", None
    feature_version: str              # "v1" from features.py
    metrics: dict[str, float]         # roc_auc, pr_auc, precision, recall …
    parameters: dict[str, str]        # hyperparams logged in MLflow
    registered_at: datetime | None = None
