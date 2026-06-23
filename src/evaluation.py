"""Point-prediction evaluation and classification reporting."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def positive_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Extract probability of encoded class 1 from an estimator with predict_proba."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("All configured estimators must expose predict_proba.")
    probabilities = np.asarray(model.predict_proba(X))
    classes = list(model.classes_)
    return probabilities[:, classes.index(1)]


def evaluate_binary_classifier(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate a fitted classifier once against an untouched test split."""
    _ = positive_probabilities(model, X_test)  # warm-up avoids first-call timing artefacts
    elapsed: list[float] = []
    probabilities: np.ndarray | None = None
    for _ in range(5):
        start = time.perf_counter()
        probabilities = positive_probabilities(model, X_test)
        elapsed.append((time.perf_counter() - start) * 1000 / len(X_test))
    assert probabilities is not None
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    metrics: dict[str, Any] = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_test, predictions, zero_division=0)),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "inference_time_ms_mean": float(np.mean(elapsed)),
        "inference_time_ms_std": float(np.std(elapsed, ddof=1)),
        "positive_prediction_rate": float(predictions.mean()),
        "predicted_probability_mean": float(probabilities.mean()),
        "predicted_probability_std": float(probabilities.std(ddof=1)),
    }
    return metrics, probabilities, predictions


def classification_report_text(y_test: pd.Series, predictions: np.ndarray) -> str:
    """Create a consistently labelled text classification report."""
    return classification_report(
        y_test, predictions, target_names=["Non-diabetic", "Diabetic"], digits=4, zero_division=0
    )


def operating_threshold_for_sensitivity(
    y_train: pd.Series, oof_probabilities: np.ndarray, target_sensitivity: float
) -> float:
    """Choose the largest OOF-derived threshold attaining a prespecified sensitivity."""
    if not 0 < target_sensitivity < 1:
        raise ValueError("Target sensitivity must be strictly between zero and one.")
    positive_probabilities = np.sort(np.asarray(oof_probabilities)[y_train.to_numpy(dtype=int) == 1])
    if len(positive_probabilities) == 0:
        raise ValueError("Threshold selection requires positive training examples.")
    index = max(0, int(np.floor((1 - target_sensitivity) * len(positive_probabilities))))
    return float(positive_probabilities[index])


def threshold_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    """Evaluate a fixed clinical operating threshold on a supplied outcome set."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {
        "threshold": float(threshold), "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision_ppv": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(specificity), "negative_predictive_value": float(npv),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "positive_prediction_rate": float(predictions.mean()),
    }


def threshold_analysis(
    y_train: pd.Series, oof_probabilities: np.ndarray, y_test: pd.Series,
    test_probabilities: np.ndarray, target_sensitivities: tuple[float, ...],
) -> pd.DataFrame:
    """Select thresholds from training-only OOF scores and evaluate once on test data."""
    rows: list[dict[str, Any]] = []
    for target in target_sensitivities:
        threshold = operating_threshold_for_sensitivity(y_train, oof_probabilities, target)
        row = threshold_metrics(y_test, test_probabilities, threshold)
        row["selection_method"] = "five-fold out-of-fold calibrated training probabilities"
        row["target_training_sensitivity"] = target
        rows.append(row)
    default = threshold_metrics(y_test, test_probabilities, 0.5)
    default["selection_method"] = "default fixed probability threshold"
    default["target_training_sensitivity"] = float("nan")
    rows.append(default)
    return pd.DataFrame(rows).sort_values("threshold", ascending=False, ignore_index=True)


def _bootstrap_statistics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Return core discrimination and classification statistics for one bootstrap sample."""
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(specificity), "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def bootstrap_point_metric_confidence_intervals(
    y_test: pd.Series, probabilities: np.ndarray, repeats: int, seed: int
) -> pd.DataFrame:
    """Compute stratified percentile bootstrap 95% CIs on the untouched test set."""
    if repeats < 100:
        raise ValueError("Use at least 100 bootstrap repeats for confidence intervals.")
    labels = y_test.to_numpy(dtype=int)
    indices_by_class = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(repeats):
        selected = np.concatenate([
            rng.choice(indices_by_class[label], size=len(indices_by_class[label]), replace=True)
            for label in (0, 1)
        ])
        values = _bootstrap_statistics(labels[selected], probabilities[selected])
        for metric, value in values.items():
            samples.setdefault(metric, []).append(value)
    estimate = _bootstrap_statistics(labels, probabilities)
    return pd.DataFrame([
        {
            "metric": metric, "estimate": estimate[metric],
            "ci_95_lower": float(np.quantile(values, 0.025)),
            "ci_95_upper": float(np.quantile(values, 0.975)), "bootstrap_repeats": repeats,
        }
        for metric, values in samples.items()
    ])
