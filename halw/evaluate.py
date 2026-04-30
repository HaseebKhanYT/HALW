"""Evaluation metrics and results.csv logging."""

import csv
import os
from datetime import datetime, timezone

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"]
RESULTS_COLUMNS = [
    "timestamp",
    "notebook",
    "dataset",
    "n_samples",
    "feature_pipeline",
    "n_features",
    "model",
    *METRIC_COLUMNS,
    "notes",
]


def evaluate(model, X_test, y_test, threshold=0.5):
    """Return dict of standard binary classification metrics."""
    y_pred_proba = model.predict(X_test).ravel()
    y_pred = (y_pred_proba > threshold).astype("int32")
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "log_loss": log_loss(y_test, y_pred_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def log_run(
    *,
    notebook,
    dataset,
    n_samples,
    feature_pipeline,
    n_features,
    model,
    metrics,
    notes="",
    path="results/results.csv",
):
    """Append one row to results.csv. Creates the file with header if missing."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    write_header = not os.path.exists(path)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notebook": notebook,
        "dataset": dataset,
        "n_samples": n_samples,
        "feature_pipeline": feature_pipeline,
        "n_features": n_features,
        "model": model,
        **{k: round(metrics[k], 4) for k in METRIC_COLUMNS if k in metrics},
        "notes": notes,
    }

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return row
