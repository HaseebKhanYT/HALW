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

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "log_loss": "Log Loss",
}
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


def format_metrics(metrics, title="Evaluation Metrics"):
    """Render metrics dict as an aligned text block with confusion matrix."""
    width = 44
    rule = "─" * width
    lines = [rule, f" {title}", rule]

    label_width = max(len(METRIC_LABELS[k]) for k in METRIC_COLUMNS if k in metrics)
    for k in METRIC_COLUMNS:
        if k in metrics:
            lines.append(f" {METRIC_LABELS[k]:<{label_width}}   {metrics[k]:.4f}")

    cm = metrics.get("confusion_matrix")
    if cm is not None:
        lines.append(rule)
        lines.append(" Confusion Matrix")
        lines.append(f" {'':<10}{'Pred 0':>10}{'Pred 1':>10}")
        for i, row in enumerate(cm):
            lines.append(f" {'Actual ' + str(i):<10}{row[0]:>10}{row[1]:>10}")
    lines.append(rule)
    return "\n".join(lines)


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
