"""Inference helpers: risk policy, fallback baseline and batch scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from ml.feature_engineering import FEATURE_COLUMNS
from utils.logging import get_logger

logger = get_logger(__name__)

RISK_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.6, "high"),
    (0.3, "medium"),
    (0.0, "low"),
)

RECOMMENDED_ACTIONS: dict[str, str] = {
    "low": "monitor normally",
    "medium": "confirm carrier capacity and notify logistics",
    "high": "prioritize fulfillment and proactively alert the customer",
}


def assign_risk_level(probability: float) -> str:
    """Map a late-delivery probability to an operational risk level."""
    for threshold, level in RISK_THRESHOLDS:
        if probability >= threshold:
            return level
    return "low"


def recommended_action(risk_level: str) -> str:
    """Return the operations playbook action for a risk level."""
    return RECOMMENDED_ACTIONS.get(risk_level, RECOMMENDED_ACTIONS["low"])


class HeuristicBaselineModel:
    """Deterministic purchase-time heuristic used when no registered model is
    available (cold start, registry outage, tests).

    Exposes the sklearn ``predict_proba`` contract over the shared feature
    columns so it is a drop-in replacement for the real pipeline.
    """

    name = "heuristic-baseline"

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Score rows with a bounded, explainable heuristic."""
        estimated_days = pd.to_numeric(X.get("estimated_days"), errors="coerce").fillna(14.0)
        freight_ratio = pd.to_numeric(X.get("freight_ratio"), errors="coerce").fillna(0.2)
        same_state = pd.to_numeric(X.get("same_state"), errors="coerce").fillna(0.0)
        seller_late_rate = pd.to_numeric(X.get("seller_late_rate"), errors="coerce").fillna(0.08)

        # Short promised windows and heavy relative freight raise risk;
        # same-state shipments and reliable sellers lower it.
        score = (
            0.05
            + 0.28 * np.clip((14.0 - estimated_days) / 14.0, 0.0, 1.0)
            + 0.20 * np.clip(freight_ratio, 0.0, 1.0)
            + 0.45 * np.clip(seller_late_rate, 0.0, 1.0)
            - 0.05 * same_state
        )
        positive = np.clip(score, 0.01, 0.98)
        return np.column_stack([1.0 - positive, positive])


def score_frame(model: object, features: pl.DataFrame) -> np.ndarray:
    """Score a Polars feature frame with any predict_proba-capable model."""
    missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise KeyError(f"Feature frame missing columns: {missing}")
    pdf = features.select(FEATURE_COLUMNS).to_pandas()
    probabilities = model.predict_proba(pdf)[:, 1]
    return np.asarray(probabilities, dtype=float)
