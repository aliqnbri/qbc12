"""Train, compare and log late-delivery model candidates with MLflow.

Trains a RandomForest and a gradient-boosting candidate (XGBoost when
installed, sklearn HistGradientBoosting otherwise) on a chronological split,
logs parameters/metrics/artifacts per run and returns the best candidate by
ROC AUC.

CLI:
    python -m ml.train --data-dir data [--register]
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import polars as pl
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from app.core.config import get_settings
from ml.cleaning import clean_datasets
from ml.evaluation import (
    evaluate_classifier,
    extract_feature_importance,
    save_confusion_matrix_artifacts,
    save_feature_importance_artifact,
)
from ml.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_full_feature_frame,
)
from ml.preprocessing import build_model_pipeline, to_model_input
from utils.logging import configure_logging, get_logger
from utils.normalize_dataset import load_datasets

logger = get_logger(__name__)

PRIMARY_METRIC = "roc_auc"


@dataclass
class CandidateResult:
    """Outcome of one trained candidate."""

    name: str
    run_id: str
    metrics: dict[str, float]


@dataclass
class TrainingResult:
    """Outcome of a full training session."""

    best: CandidateResult
    candidates: list[CandidateResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_model": self.best.name,
            "best_run_id": self.best.run_id,
            "best_metrics": self.best.metrics,
            "candidates": [
                {"name": c.name, "run_id": c.run_id, "metrics": c.metrics}
                for c in self.candidates
            ],
        }


def build_candidates(random_seed: int) -> dict[str, Any]:
    """Return the estimator candidates keyed by name.

    XGBoost is preferred as the boosting candidate; HistGradientBoosting is
    the drop-in replacement when xgboost is not installed.
    """
    candidates: dict[str, Any] = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=16,
            min_samples_leaf=5,
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_seed,
        )
    }
    try:
        from xgboost import XGBClassifier

        candidates["xgboost"] = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_seed,
        )
    except ImportError:
        logger.warning("xgboost not installed; using HistGradientBoostingClassifier")
        candidates["hist_gradient_boosting"] = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.08,
            max_depth=None,
            l2_regularization=0.1,
            random_state=random_seed,
        )
    return candidates


def _fit_and_log(
    name: str,
    pipeline: Pipeline,
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    y_test: np.ndarray,
    threshold: float,
) -> CandidateResult:
    """Fit one candidate inside an MLflow run and log everything."""
    with mlflow.start_run(run_name=name) as run:
        model = pipeline.named_steps["model"]
        params = {
            key: value
            for key, value in model.get_params(deep=False).items()
            if isinstance(value, (int, float, str, bool, type(None)))
        }
        mlflow.log_params({f"model__{key}": value for key, value in params.items()})
        mlflow.log_params(
            {
                "candidate": name,
                "n_train_rows": len(X_train),
                "n_test_rows": len(X_test),
                "n_numeric_features": len(NUMERIC_FEATURES),
                "n_categorical_features": len(CATEGORICAL_FEATURES),
                "decision_threshold": threshold,
                "split_strategy": "temporal_80_20",
            }
        )

        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate_classifier(y_test, y_prob, threshold=threshold)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for artifact in save_confusion_matrix_artifacts(metrics, tmp_path):
                mlflow.log_artifact(str(artifact))
            importance = extract_feature_importance(pipeline, X_test, y_test)
            mlflow.log_artifact(
                str(save_feature_importance_artifact(importance, tmp_path))
            )

        signature = infer_signature(X_train.head(50), y_prob[:50])
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(5),
        )
        logger.info(
            "Candidate %-24s roc_auc=%.4f pr_auc=%.4f f1=%.4f",
            name,
            metrics["roc_auc"],
            metrics["pr_auc"],
            metrics["f1"],
        )
        return CandidateResult(name=name, run_id=run.info.run_id, metrics=metrics)


def run_training(
    data_dir: str | Path | None = None,
    datasets: dict[str, pl.DataFrame] | None = None,
    register: bool = False,
) -> TrainingResult:
    """Run the full training session: clean, engineer, fit, compare, log.

    Args:
        data_dir: Directory of raw CSVs (defaults to ``settings.data_dir``).
        datasets: Pre-loaded datasets; overrides ``data_dir`` when given.
        register: Also register and promote the best model.

    Returns:
        The training result with the best candidate.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    if datasets is None:
        datasets = load_datasets(data_dir or settings.data_dir)
    datasets = clean_datasets(datasets)
    train_frame, test_frame = build_full_feature_frame(
        datasets, test_size=settings.test_size
    )

    X_train = to_model_input(train_frame)
    y_train = train_frame[TARGET_COLUMN].to_numpy()
    X_test = to_model_input(test_frame)
    y_test = test_frame[TARGET_COLUMN].to_numpy()

    results: list[CandidateResult] = []
    for name, estimator in build_candidates(settings.random_seed).items():
        pipeline = build_model_pipeline(estimator)
        results.append(
            _fit_and_log(
                name,
                pipeline,
                X_train,
                y_train,
                X_test,
                y_test,
                settings.decision_threshold,
            )
        )

    best = max(results, key=lambda result: result.metrics[PRIMARY_METRIC])
    logger.info(
        "Best candidate: %s (%s=%.4f, run_id=%s)",
        best.name,
        PRIMARY_METRIC,
        best.metrics[PRIMARY_METRIC],
        best.run_id,
    )

    if register:
        from ml.registry import register_best_model

        register_best_model(best.run_id)

    return TrainingResult(best=best, candidates=results)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train late-delivery models")
    parser.add_argument("--data-dir", default=None, help="Directory of raw CSV files")
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register and promote the best model in the MLflow registry",
    )
    args = parser.parse_args()
    result = run_training(data_dir=args.data_dir, register=args.register)
    logger.info("Training complete: %s", result.as_dict())


if __name__ == "__main__":
    main()
