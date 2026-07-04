"""Sklearn preprocessing pipeline shared by every model candidate."""

from __future__ import annotations

import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.feature_engineering import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build the column transformer applied before any estimator.

    Numeric features: median imputation + standard scaling.
    Categorical features: constant imputation + one-hot with unknown handling,
    so unseen states/categories at serving time never fail.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(estimator: object) -> Pipeline:
    """Wrap an estimator with the shared preprocessor into a single pipeline
    so preprocessing is persisted with the model in MLflow."""
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", estimator)])


def to_model_input(frame: pl.DataFrame) -> "object":
    """Project a Polars frame onto the model feature columns and convert to
    pandas, the interchange format expected by sklearn's ColumnTransformer."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"Feature frame is missing columns: {missing}")
    return frame.select(FEATURE_COLUMNS).to_pandas()
