"""Correct finite-sample split conformal classification utilities."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def class_probability_matrix(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return probabilities in fixed encoded-class order [0, 1]."""
    raw = np.asarray(model.predict_proba(X))
    output = np.zeros((len(X), 2), dtype=float)
    for index, label in enumerate(model.classes_):
        output[:, int(label)] = raw[:, index]
    return output


def conformal_quantile(calibration_probabilities: np.ndarray, y_calib: Iterable[int], confidence: float) -> float:
    """Calculate split-conformal q-hat for score ``1 - p(true class)``."""
    labels = np.asarray(list(y_calib), dtype=int)
    if len(labels) == 0:
        raise ValueError("Calibration set is empty.")
    scores = np.sort(1.0 - calibration_probabilities[np.arange(len(labels)), labels])
    k = math.ceil((len(labels) + 1) * confidence)
    return float(scores[k - 1]) if k <= len(labels) else float("inf")


def prediction_sets(probabilities: np.ndarray, q_hat: float) -> np.ndarray:
    """Create boolean class sets: class c is included iff 1-p(c) <= q-hat."""
    return (1.0 - probabilities) <= q_hat


def _per_class_metric(values: np.ndarray, labels: np.ndarray, target_class: int) -> tuple[float, float]:
    mask = labels == target_class
    if not mask.any():
        return float("nan"), float("nan")
    return float(values[mask, target_class].mean()), float(values[mask].sum(axis=1).mean())


def evaluate_conformal(
    calibration_probabilities: np.ndarray,
    y_calib: pd.Series,
    test_probabilities: np.ndarray,
    y_test: pd.Series,
    confidence_levels: Iterable[float],
) -> tuple[pd.DataFrame, dict[float, np.ndarray], dict[float, float]]:
    """Evaluate true-label coverage and set efficiency across confidence levels."""
    labels = y_test.to_numpy(dtype=int)
    rows: list[dict[str, float | int]] = []
    sets: dict[float, np.ndarray] = {}
    quantiles: dict[float, float] = {}
    for confidence in confidence_levels:
        q_hat = conformal_quantile(calibration_probabilities, y_calib, confidence)
        sets_at_confidence = prediction_sets(test_probabilities, q_hat)
        sizes = sets_at_confidence.sum(axis=1)
        covered = sets_at_confidence[np.arange(len(labels)), labels]
        class_0_coverage, class_0_size = _per_class_metric(sets_at_confidence, labels, 0)
        class_1_coverage, class_1_size = _per_class_metric(sets_at_confidence, labels, 1)
        sets[float(confidence)] = sets_at_confidence
        quantiles[float(confidence)] = q_hat
        rows.append({
            "confidence_level": float(confidence), "alpha": float(1 - confidence), "q_hat": q_hat,
            "calibration_size": int(len(y_calib)), "test_size": int(len(y_test)),
            "empirical_coverage": float(covered.mean()), "coverage_gap": float(covered.mean() - confidence),
            "mean_set_size": float(sizes.mean()), "median_set_size": float(np.median(sizes)),
            "singleton_rate": float((sizes == 1).mean()), "doubleton_rate": float((sizes == 2).mean()),
            "empty_set_rate": float((sizes == 0).mean()), "class_0_coverage": class_0_coverage,
            "class_1_coverage": class_1_coverage, "class_0_mean_set_size": class_0_size,
            "class_1_mean_set_size": class_1_size,
        })
    return pd.DataFrame(rows), sets, quantiles


def prediction_preview(
    test_probabilities: np.ndarray,
    y_test: pd.Series,
    sets_at_90: np.ndarray,
    sample_ids: pd.Index,
) -> pd.DataFrame:
    """Return first 20 and 20 least-confident test observations without duplicate rows."""
    point_predictions = test_probabilities.argmax(axis=1)
    confidence = test_probabilities.max(axis=1)
    order = list(range(min(20, len(y_test)))) + list(np.argsort(confidence)[:20])
    selected = list(dict.fromkeys(order))
    labels = y_test.to_numpy(dtype=int)
    rows = []
    for position in selected:
        included = [str(label) for label, present in enumerate(sets_at_90[position]) if present]
        rows.append({
            "sample_id": str(sample_ids[position]), "true_label": int(labels[position]),
            "p_class_0": float(test_probabilities[position, 0]), "p_class_1": float(test_probabilities[position, 1]),
            "point_prediction": int(point_predictions[position]), "conformal_set_90": "{" + ", ".join(included) + "}",
            "set_size_90": int(sets_at_90[position].sum()),
            "is_covered_90": bool(sets_at_90[position, labels[position]]),
        })
    return pd.DataFrame(rows)


def calibration_size_sensitivity(
    calibration_probabilities: np.ndarray,
    y_calib: pd.Series,
    test_probabilities: np.ndarray,
    y_test: pd.Series,
    fractions: Iterable[float],
    confidence_levels: Iterable[float],
    seed: int,
    repeats: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat stratified calibration subsampling while holding the test set fixed."""
    labels = y_test.to_numpy(dtype=int)
    rows: list[dict[str, float | int]] = []
    indices = np.arange(len(y_calib))
    for fraction in fractions:
        for repeat in range(repeats):
            if fraction >= 1.0:
                selected = indices
            else:
                selected, _ = train_test_split(
                    indices, train_size=fraction, stratify=y_calib, random_state=seed + repeat
                )
            for confidence in confidence_levels:
                q_hat = conformal_quantile(calibration_probabilities[selected], y_calib.iloc[selected], confidence)
                sets = prediction_sets(test_probabilities, q_hat)
                sizes = sets.sum(axis=1)
                covered = sets[np.arange(len(labels)), labels]
                class_0_coverage, _ = _per_class_metric(sets, labels, 0)
                class_1_coverage, _ = _per_class_metric(sets, labels, 1)
                rows.append({
                    "calibration_fraction": float(fraction), "repeat": repeat + 1,
                    "calibration_size": int(len(selected)), "confidence_level": float(confidence),
                    "q_hat": q_hat, "empirical_coverage": float(covered.mean()),
                    "mean_set_size": float(sizes.mean()), "class_0_coverage": class_0_coverage,
                    "class_1_coverage": class_1_coverage,
                })
    detailed = pd.DataFrame(rows)
    summary = detailed.groupby(["calibration_fraction", "confidence_level"], as_index=False).agg(
        calibration_size_mean=("calibration_size", "mean"), coverage_mean=("empirical_coverage", "mean"),
        coverage_std=("empirical_coverage", "std"), coverage_min=("empirical_coverage", "min"),
        coverage_max=("empirical_coverage", "max"), mean_set_size_mean=("mean_set_size", "mean"),
        mean_set_size_std=("mean_set_size", "std"), class_0_coverage_mean=("class_0_coverage", "mean"),
        class_1_coverage_mean=("class_1_coverage", "mean"), repeats=("repeat", "count"),
    )
    return detailed, summary


def mondrian_quantiles(
    calibration_probabilities: np.ndarray, y_calib: pd.Series, confidence: float
) -> np.ndarray:
    """Calculate one finite-sample split-conformal threshold per true class."""
    labels = y_calib.to_numpy(dtype=int)
    thresholds = np.empty(2, dtype=float)
    for label in (0, 1):
        mask = labels == label
        if not mask.any():
            raise ValueError(f"Mondrian conformal calibration has no examples for class {label}.")
        scores = np.sort(1.0 - calibration_probabilities[mask, label])
        k = math.ceil((len(scores) + 1) * confidence)
        thresholds[label] = scores[k - 1] if k <= len(scores) else float("inf")
    return thresholds


def evaluate_mondrian_conformal(
    calibration_probabilities: np.ndarray,
    y_calib: pd.Series,
    test_probabilities: np.ndarray,
    y_test: pd.Series,
    confidence_levels: Iterable[float],
) -> tuple[pd.DataFrame, dict[float, np.ndarray], dict[float, np.ndarray]]:
    """Evaluate class-conditional (Mondrian) split conformal prediction sets."""
    labels = y_test.to_numpy(dtype=int)
    rows: list[dict[str, float | int]] = []
    sets_by_confidence: dict[float, np.ndarray] = {}
    thresholds_by_confidence: dict[float, np.ndarray] = {}
    for confidence in confidence_levels:
        thresholds = mondrian_quantiles(calibration_probabilities, y_calib, confidence)
        sets = (1.0 - test_probabilities) <= thresholds[np.newaxis, :]
        sizes = sets.sum(axis=1)
        covered = sets[np.arange(len(labels)), labels]
        class_0_coverage, class_0_size = _per_class_metric(sets, labels, 0)
        class_1_coverage, class_1_size = _per_class_metric(sets, labels, 1)
        confidence = float(confidence)
        sets_by_confidence[confidence] = sets
        thresholds_by_confidence[confidence] = thresholds
        rows.append({
            "confidence_level": confidence, "alpha": float(1 - confidence),
            "q_hat_class_0": float(thresholds[0]), "q_hat_class_1": float(thresholds[1]),
            "calibration_size": int(len(y_calib)), "test_size": int(len(y_test)),
            "empirical_coverage": float(covered.mean()), "coverage_gap": float(covered.mean() - confidence),
            "mean_set_size": float(sizes.mean()), "median_set_size": float(np.median(sizes)),
            "singleton_rate": float((sizes == 1).mean()), "doubleton_rate": float((sizes == 2).mean()),
            "empty_set_rate": float((sizes == 0).mean()), "class_0_coverage": class_0_coverage,
            "class_1_coverage": class_1_coverage, "class_0_mean_set_size": class_0_size,
            "class_1_mean_set_size": class_1_size,
        })
    return pd.DataFrame(rows), sets_by_confidence, thresholds_by_confidence


def bootstrap_conformal_confidence_intervals(
    y_test: pd.Series, sets_by_confidence: dict[float, np.ndarray], method: str, repeats: int, seed: int
) -> pd.DataFrame:
    """Compute stratified percentile bootstrap CIs for fixed test-set conformal outputs."""
    if repeats < 100:
        raise ValueError("Use at least 100 bootstrap repeats for confidence intervals.")
    labels = y_test.to_numpy(dtype=int)
    indices_by_class = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for confidence, sets in sets_by_confidence.items():
        sampled: dict[str, list[float]] = {"empirical_coverage": [], "class_0_coverage": [], "class_1_coverage": [], "mean_set_size": []}
        for _ in range(repeats):
            selected = np.concatenate([
                rng.choice(indices_by_class[label], size=len(indices_by_class[label]), replace=True)
                for label in (0, 1)
            ])
            sampled_sets = sets[selected]
            sampled_labels = labels[selected]
            sampled["empirical_coverage"].append(float(sampled_sets[np.arange(len(selected)), sampled_labels].mean()))
            sampled["class_0_coverage"].append(float(sampled_sets[sampled_labels == 0, 0].mean()))
            sampled["class_1_coverage"].append(float(sampled_sets[sampled_labels == 1, 1].mean()))
            sampled["mean_set_size"].append(float(sampled_sets.sum(axis=1).mean()))
        point = {
            "empirical_coverage": float(sets[np.arange(len(labels)), labels].mean()),
            "class_0_coverage": float(sets[labels == 0, 0].mean()),
            "class_1_coverage": float(sets[labels == 1, 1].mean()),
            "mean_set_size": float(sets.sum(axis=1).mean()),
        }
        for metric, values in sampled.items():
            rows.append({
                "method": method, "confidence_level": confidence, "metric": metric,
                "estimate": point[metric], "ci_95_lower": float(np.quantile(values, 0.025)),
                "ci_95_upper": float(np.quantile(values, 0.975)), "bootstrap_repeats": repeats,
            })
    return pd.DataFrame(rows)
