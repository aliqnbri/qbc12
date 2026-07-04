"""Feature engineering tests: target correctness and leakage safety."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from ml.feature_engineering import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FORBIDDEN_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_full_feature_frame,
    build_inference_frame,
    build_target,
    build_training_frame,
    compute_seller_history,
    temporal_train_test_split,
)


def test_no_forbidden_feature_columns() -> None:
    """The model must never see outcome-derived columns."""
    leaked = FORBIDDEN_COLUMNS.intersection(FEATURE_COLUMNS)
    assert not leaked, f"Leaky features detected: {sorted(leaked)}"


def test_feature_lists_are_disjoint_and_complete() -> None:
    assert not set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)
    assert set(FEATURE_COLUMNS) == set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)


def test_build_target(synthetic_datasets: dict[str, pl.DataFrame]) -> None:
    labelled = build_target(synthetic_datasets["orders"])
    # The shipped (undelivered) order must be excluded from the label set.
    assert labelled.height == 6
    by_id = {row["order_id"]: row[TARGET_COLUMN] for row in labelled.to_dicts()}
    assert by_id == {"o0": 0, "o1": 1, "o2": 0, "o3": 1, "o4": 0, "o5": 1}


def test_build_training_frame(synthetic_datasets: dict[str, pl.DataFrame]) -> None:
    frame = build_training_frame(synthetic_datasets)
    assert frame.height == 6
    for column in ("order_id", "order_purchase_timestamp", TARGET_COLUMN):
        assert column in frame.columns
    # Every non-history feature must be present.
    history = {
        "seller_order_count",
        "seller_late_rate",
        "seller_avg_delay_days",
        "customer_order_count",
    }
    for column in set(FEATURE_COLUMNS) - history:
        assert column in frame.columns, f"missing feature {column}"
    # Estimated days must reflect the 14-day promise window.
    assert frame["estimated_days"].min() > 13.0
    assert frame["estimated_days"].max() < 15.0
    # Categories must be translated to English.
    assert set(frame["product_category"].unique().to_list()) <= {
        "electronics",
        "furniture_decor",
    }


def test_temporal_split_is_chronological(
    synthetic_datasets: dict[str, pl.DataFrame]
) -> None:
    frame = build_training_frame(synthetic_datasets)
    train, test = temporal_train_test_split(frame, test_size=0.34)
    assert train.height + test.height == frame.height
    assert (
        train["order_purchase_timestamp"].max()
        <= test["order_purchase_timestamp"].min()
    )


def test_seller_history_from_train_window_only(
    synthetic_datasets: dict[str, pl.DataFrame]
) -> None:
    train_frame, test_frame = build_full_feature_frame(
        synthetic_datasets, test_size=0.34
    )
    assert "seller_late_rate" in train_frame.columns
    assert "seller_late_rate" in test_frame.columns
    history = compute_seller_history(
        temporal_train_test_split(
            build_training_frame(synthetic_datasets), test_size=0.34
        )[0]
    )
    rates = dict(zip(history["seller_id"].to_list(), history["seller_late_rate"].to_list()))
    assert all(0.0 <= rate <= 1.0 for rate in rates.values())


def test_build_inference_frame_schema() -> None:
    frame = build_inference_frame(
        [
            {
                "order_id": "x-1",
                "order_purchase_timestamp": datetime(2018, 6, 1, 14, 30),
                "order_estimated_delivery_date": datetime(2018, 6, 20),
                "shipping_limit_date": None,
                "customer_state": "SP",
                "customer_zip_code_prefix": "01310",
                "seller_state": "RJ",
                "seller_zip_code_prefix": "20040",
                "price": 100.0,
                "freight_value": 12.0,
                "item_count": 2,
            }
        ]
    )
    assert frame.columns == FEATURE_COLUMNS
    assert frame.height == 1
    row = frame.to_dicts()[0]
    assert 18.0 < row["estimated_days"] < 19.0
    assert row["purchase_hour"] == 14.0
    assert row["same_state"] == 0.0
    assert row["zip_distance"] == abs(1310 - 20040)
    # Optional fields default to null for the imputer.
    assert row["total_weight_g"] is None
    assert row["seller_late_rate"] is None
