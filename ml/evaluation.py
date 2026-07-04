"""Model evaluation: metrics, confusion matrix and feature importance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from utils.logging import get_logger

logger = get_logger(__name__)


def evaluate_classifier(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute the full metric suite for a binary classifier.

    Returns ROC AUC, PR AUC, precision, recall, F1 and the confusion-matrix
    cells (tn/fp/fn/tp) as flat floats, ready for MLflow logging.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negatives": float(tn),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "true_positives": float(tp),
    }


def save_confusion_matrix_artifacts(
    metrics: dict[str, float], output_dir: Path
) -> list[Path]:
    """Persist the confusion matrix as JSON and as a PNG heatmap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = [
        [metrics["true_negatives"], metrics["false_positives"]],
        [metrics["false_negatives"], metrics["true_positives"]],
    ]
    json_path = output_dir / "confusion_matrix.json"
    json_path.write_text(
        json.dumps({"labels": ["on_time", "late"], "matrix": matrix}, indent=2)
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4.5, 4))
    image = axis.imshow(np.array(matrix), cmap="Blues")
    axis.set_xticks([0, 1], labels=["on_time", "late"])
    axis.set_yticks([0, 1], labels=["on_time", "late"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Confusion matrix")
    for row in range(2):
        for column in range(2):
            axis.text(
                column, row, f"{int(matrix[row][column])}", ha="center", va="center"
            )
    figure.colorbar(image)
    png_path = output_dir / "confusion_matrix.png"
    figure.tight_layout()
    figure.savefig(png_path, dpi=120)
    plt.close(figure)
    return [json_path, png_path]


def extract_feature_importance(
    pipeline: Pipeline,
    X_valid: Any,
    y_valid: np.ndarray,
    max_permutation_rows: int = 2000,
    random_state: int = 42,
) -> dict[str, float]:
    """Extract per-feature importance from a fitted pipeline.

    Uses the estimator's native ``feature_importances_`` on the transformed
    feature space when available; otherwise falls back to permutation
    importance on the raw features (subsampled for speed).
    """
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    if hasattr(model, "feature_importances_"):
        names = list(preprocessor.get_feature_names_out())
        importances = model.feature_importances_
        pairs = sorted(zip(names, importances), key=lambda item: item[1], reverse=True)
        return {name: float(value) for name, value in pairs}

    sample = X_valid
    y_sample = y_valid
    if len(X_valid) > max_permutation_rows:
        sample = X_valid.sample(max_permutation_rows, random_state=random_state)
        y_sample = y_valid[sample.index]
    result = permutation_importance(
        pipeline, sample, y_sample, n_repeats=3, random_state=random_state, scoring="roc_auc"
    )
    pairs = sorted(
        zip(list(sample.columns), result.importances_mean),
        key=lambda item: item[1],
        reverse=True,
    )
    return {name: float(value) for name, value in pairs}


def save_feature_importance_artifact(
    importance: dict[str, float], output_dir: Path, top_n: int = 30
) -> Path:
    """Persist feature importances as a JSON artifact (top N)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    top = dict(list(importance.items())[:top_n])
    path = output_dir / "feature_importance.json"
    path.write_text(json.dumps(top, indent=2))
    return path
