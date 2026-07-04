"""Shared fixtures: hermetic API client, synthetic datasets, sample payloads."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

import polars as pl
import pytest
from fastapi.testclient import TestClient

# Hermetic settings: unreachable registry, no local artifact, no DB logging.
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
os.environ.setdefault("MODEL_LOCAL_PATH", "/nonexistent/model.joblib")
os.environ.setdefault("ENABLE_DB_LOGGING", "false")

from app.main import app  # noqa: E402
from app.services.model_service import get_model_service  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    """API test client backed by the deterministic heuristic fallback model."""
    service = get_model_service()
    service._use_fallback()
    return TestClient(app)


@pytest.fixture()
def sample_payload() -> dict[str, Any]:
    """A valid purchase-time prediction payload."""
    return {
        "order_id": "test-order-0001",
        "order_purchase_timestamp": "2018-06-01T14:30:00",
        "order_estimated_delivery_date": "2018-06-20T00:00:00",
        "shipping_limit_date": "2018-06-05T00:00:00",
        "customer_state": "SP",
        "customer_zip_code_prefix": "01310",
        "seller_state": "RJ",
        "seller_zip_code_prefix": "20040",
        "price": 129.9,
        "freight_value": 19.9,
        "item_count": 2,
        "distinct_sellers": 1,
        "payment_type": "credit_card",
        "payment_installments": 3,
        "payment_value": 149.8,
        "product_category": "electronics",
        "product_weight_g": 800,
        "product_length_cm": 30,
        "product_height_cm": 10,
        "product_width_cm": 20,
        "product_photos_qty": 3,
        "product_description_length": 500,
    }


@pytest.fixture()
def synthetic_datasets() -> dict[str, pl.DataFrame]:
    """A tiny but structurally complete set of raw Olist datasets."""
    base = datetime(2018, 1, 1, 10, 0, 0)

    def order_row(
        index: int, late: bool, status: str = "delivered"
    ) -> dict[str, Any]:
        purchase = base + timedelta(days=index)
        estimated = purchase + timedelta(days=14)
        delivered = estimated + (timedelta(days=3) if late else -timedelta(days=2))
        return {
            "order_id": f"o{index}",
            "customer_id": f"c{index}",
            "order_status": status,
            "order_purchase_timestamp": purchase,
            "order_approved_at": purchase + timedelta(hours=1),
            "order_delivered_carrier_date": purchase + timedelta(days=2),
            "order_delivered_customer_date": delivered if status == "delivered" else None,
            "order_estimated_delivery_date": estimated,
        }

    orders = pl.DataFrame(
        [
            order_row(0, late=False),
            order_row(1, late=True),
            order_row(2, late=False),
            order_row(3, late=True),
            order_row(4, late=False),
            order_row(5, late=True),
            order_row(6, late=False, status="shipped"),
        ]
    )

    order_items = pl.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(6)],
            "order_item_id": [1] * 6,
            "product_id": [f"p{i % 2}" for i in range(6)],
            "seller_id": [f"s{i % 2}" for i in range(6)],
            "shipping_limit_date": [
                base + timedelta(days=i, hours=72) for i in range(6)
            ],
            "price": [100.0, 50.0, 200.0, 80.0, 120.0, 60.0],
            "freight_value": [10.0, 8.0, 25.0, 12.0, 15.0, 9.0],
        }
    )

    order_payments = pl.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(6)],
            "payment_sequential": [1] * 6,
            "payment_type": ["credit_card", "boleto"] * 3,
            "payment_installments": [1, 2, 3, 1, 4, 1],
            "payment_value": [110.0, 58.0, 225.0, 92.0, 135.0, 69.0],
        }
    )

    customers = pl.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(7)],
            "customer_unique_id": [f"u{i % 3}" for i in range(7)],
            "customer_zip_code_prefix": ["01310"] * 7,
            "customer_city": ["sao paulo"] * 7,
            "customer_state": ["SP", "RJ", "SP", "MG", "SP", "RJ", "SP"],
        }
    )

    products = pl.DataFrame(
        {
            "product_id": ["p0", "p1"],
            "product_category_name": ["eletronicos", "moveis_decoracao"],
            "product_name_length": [40, 55],
            "product_description_length": [300, 800],
            "product_photos_qty": [2, 5],
            "product_weight_g": [500.0, 4000.0],
            "product_length_cm": [20.0, 60.0],
            "product_height_cm": [10.0, 40.0],
            "product_width_cm": [15.0, 50.0],
        }
    )

    sellers = pl.DataFrame(
        {
            "seller_id": ["s0", "s1"],
            "seller_zip_code_prefix": ["20040", "30110"],
            "seller_city": ["rio de janeiro", "belo horizonte"],
            "seller_state": ["RJ", "MG"],
        }
    )

    translation = pl.DataFrame(
        {
            "product_category_name": ["eletronicos", "moveis_decoracao"],
            "product_category_name_english": ["electronics", "furniture_decor"],
        }
    )

    return {
        "orders": orders,
        "order_items": order_items,
        "order_payments": order_payments,
        "customers": customers,
        "products": products,
        "sellers": sellers,
        "product_category_name_translation": translation,
    }
