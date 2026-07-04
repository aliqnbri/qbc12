"""Preprocessing pipeline tests: imputation, encoding, unknown handling."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.cleaning import iqr_cap
from ml.feature_engineering import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from ml.preprocessing import build_model_pipeline, build_preprocessor, to_model_input


def _feature_frame(rows: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, list] = {}
    for column in NUMERIC_FEATURES:
        values = rng.normal(10.0, 3.0, rows).tolist()
        values[0] = None  # every numeric column carries a null to impute
        data[column] = values
    states = ["SP", "RJ", "MG", None]
    for column in CATEGORICAL_FEATURES:
        data[column] = [states[i % len(states)] for i in range(rows)]
    return pl.DataFrame(data, schema_overrides={c: pl.Float64 for c in NUMERIC_FEATURES})


def test_preprocessor_handles_nulls_and_unknowns() -> None:
    train = _feature_frame(60, seed=1)
    preprocessor = build_preprocessor()
    X_train = to_model_input(train)
    transformed = preprocessor.fit_transform(X_train)
    assert transformed.shape[0] == 60

    unseen = _feature_frame(5, seed=2).with_columns(
        pl.lit("ZZ").alias("customer_state")  # category never seen in training
    )
    transformed_unseen = preprocessor.transform(to_model_input(unseen))
    assert transformed_unseen.shape[0] == 5
    assert not np.isnan(
        transformed_unseen.toarray()
        if hasattr(transformed_unseen, "toarray")
        else transformed_unseen
    ).any()


def test_model_pipeline_fit_predict() -> None:
    frame = _feature_frame(80, seed=3)
    y = (np.arange(80) % 2).astype(int)
    pipeline = build_model_pipeline(
        RandomForestClassifier(n_estimators=10, random_state=0)
    )
    pipeline.fit(to_model_input(frame), y)
    probabilities = pipeline.predict_proba(to_model_input(frame))
    assert probabilities.shape == (80, 2)
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)


def test_to_model_input_requires_all_features() -> None:
    incomplete = pl.DataFrame({FEATURE_COLUMNS[0]: [1.0]})
    with pytest.raises(KeyError):
        to_model_input(incomplete)


def test_iqr_cap_bounds_outliers() -> None:
    frame = pl.DataFrame({"price": [10.0, 12.0, 11.0, 13.0, 9.0, 10_000.0, None]})
    capped = iqr_cap(frame, "price")
    assert capped["price"].max() < 10_000.0
    assert capped.height == frame.height  # rows preserved, not dropped
    assert capped["price"].null_count() == 1  # nulls preserved for the imputer
