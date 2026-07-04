"""Model service and fallback model tests."""

from __future__ import annotations

import numpy as np

from app.services.model_service import SOURCE_FALLBACK, ModelService
from ml.feature_engineering import FEATURE_COLUMNS, build_inference_frame
from ml.predict import HeuristicBaselineModel, assign_risk_level, recommended_action
from ml.preprocessing import to_model_input


def _inference_input(estimated_days: float) -> "object":
    from datetime import datetime, timedelta

    purchase = datetime(2018, 6, 1, 12, 0)
    frame = build_inference_frame(
        [
            {
                "order_id": "m-1",
                "order_purchase_timestamp": purchase,
                "order_estimated_delivery_date": purchase
                + timedelta(days=estimated_days),
                "customer_state": "SP",
                "seller_state": "SP",
                "price": 100.0,
                "freight_value": 10.0,
                "item_count": 1,
            }
        ]
    )
    return to_model_input(frame)


def test_fallback_model_loads_when_registry_unavailable() -> None:
    service = ModelService()
    service.load()  # registry unreachable + no local artifact in the test env
    assert service.is_loaded
    assert service.source == SOURCE_FALLBACK
    assert service.model_name == HeuristicBaselineModel.name


def test_fallback_predict_proba_contract() -> None:
    model = HeuristicBaselineModel()
    probabilities = model.predict_proba(_inference_input(14.0))
    assert probabilities.shape == (1, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert 0.0 <= probabilities[0, 1] <= 1.0


def test_fallback_risk_increases_with_shorter_promise() -> None:
    model = HeuristicBaselineModel()
    tight = model.predict_proba(_inference_input(3.0))[0, 1]
    generous = model.predict_proba(_inference_input(30.0))[0, 1]
    assert tight > generous


def test_service_predict_proba_shape() -> None:
    service = ModelService()
    service._use_fallback()
    probabilities = service.predict_proba(_inference_input(10.0))
    assert probabilities.shape == (1,)
    assert 0.0 <= probabilities[0] <= 1.0
    assert service.feature_count == len(FEATURE_COLUMNS)


def test_risk_policy() -> None:
    assert assign_risk_level(0.05) == "low"
    assert assign_risk_level(0.3) == "medium"
    assert assign_risk_level(0.59) == "medium"
    assert assign_risk_level(0.6) == "high"
    assert assign_risk_level(0.99) == "high"
    for level in ("low", "medium", "high"):
        assert recommended_action(level)
