"""API contract tests against the deterministic fallback model."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

REQUIRED_PREDICTION_KEYS = {
    "order_id",
    "late_probability",
    "late_delivery_probability",
    "risk_level",
    "model_name",
    "model_version",
    "recommended_action",
    "latency_seconds",
}


def test_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["docs_url"] == "/docs"
    assert body["health_url"] == "/health"


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client: TestClient) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"]
    assert body["feature_count"] > 0
    assert "purchase-time" in body["temporal_leakage_policy"]


def test_predict_contract(client: TestClient, sample_payload: dict[str, Any]) -> None:
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    body = response.json()
    assert REQUIRED_PREDICTION_KEYS <= set(body)
    assert body["order_id"] == sample_payload["order_id"]
    assert 0.0 <= body["late_probability"] <= 1.0
    assert body["late_probability"] == body["late_delivery_probability"]
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["recommended_action"]


def test_predict_is_deterministic(
    client: TestClient, sample_payload: dict[str, Any]
) -> None:
    first = client.post("/predict", json=sample_payload).json()
    second = client.post("/predict", json=sample_payload).json()
    assert first["late_probability"] == second["late_probability"]


def test_predict_batch(client: TestClient, sample_payload: dict[str, Any]) -> None:
    other = dict(sample_payload, order_id="test-order-0002", price=45.0)
    response = client.post("/predict-batch", json={"orders": [sample_payload, other]})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert {p["order_id"] for p in body["predictions"]} == {
        "test-order-0001",
        "test-order-0002",
    }


def test_batch_predict_alias(client: TestClient, sample_payload: dict[str, Any]) -> None:
    response = client.post("/batch-predict", json={"orders": [sample_payload]})
    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_predict_rejects_invalid_dates(
    client: TestClient, sample_payload: dict[str, Any]
) -> None:
    payload = dict(sample_payload)
    payload["order_estimated_delivery_date"] = "2018-05-01T00:00:00"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_leakage_fields(
    client: TestClient, sample_payload: dict[str, Any]
) -> None:
    payload = dict(sample_payload, review_score=1)
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_metrics_exposition(client: TestClient, sample_payload: dict[str, Any]) -> None:
    client.post("/predict", json=sample_payload)
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert "olist_predictions_total" in text
    assert "olist_request_latency_seconds" in text


def test_metrics_summary(client: TestClient, sample_payload: dict[str, Any]) -> None:
    client.post("/predict", json=sample_payload)
    response = client.get("/metrics-summary")
    assert response.status_code == 200
    body = response.json()
    assert body["total_predictions"] >= 1
    assert isinstance(body["risk_distribution"], dict)


def test_docs_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
