#!/usr/bin/env python3
"""Publication-ready reviewer-response experiment.

This script keeps the original full experiment intact and writes a new output
folder with reviewer-response analyses:

1. exact source/target split tables,
2. leakage-safe source calibration selection,
3. target-calibration-only adaptive selection,
4. shift-aware baselines,
5. target calibration-size ablation, and
6. equal-budget source-validation hyperparameter tuning.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

matplotlib_cache = Path(tempfile.gettempdir()) / "adaptive_transport_calibration_mpl"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import run_full_adaptive_transport_5models as base


TARGET = base.TARGET
PRIMARY_CONFIDENCE = base.PRIMARY_CONFIDENCE
CALIBRATION_METHODS = base.CALIBRATION_METHODS
FEATURE_SETS = base.FEATURE_SETS
ATCP_SOURCE_WEIGHT_GRID = (0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00)
EVAL_CONFIDENCE_LEVELS = (0.80, 0.85, 0.90, 0.92, 0.95)
DP_CLASS0_CONFIDENCE_GRID = (0.85, 0.87, 0.88, 0.89, 0.90)
DP_CLASS1_CONFIDENCE_GRID = (0.90, 0.92, 0.94, 0.95, 0.97)
SOURCE_TRUST_MIX_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
RISK_ASYMMETRY_DELTA = 0.02


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Reviewer-response publication-ready adaptive transport experiment")
    parser.add_argument("--input-dir", default=str(root / "outputs_full_5models"))
    parser.add_argument("--outdir", default=str(root / "outputs_publication_ready_review_response"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=500)
    parser.add_argument("--tuning-candidates", type=int, default=3)
    parser.add_argument("--ablation-repeats", type=int, default=20)
    parser.add_argument("--bootstrap-repeats", type=int, default=300)
    parser.add_argument("--quick", action="store_true", help="Use lighter estimators for a faster smoke run.")
    return parser.parse_args()


def ensure_tree(outdir: Path) -> dict[str, Path]:
    paths = {
        "root": outdir,
        "data": outdir / "data",
        "tables": outdir / "tables",
        "figures": outdir / "figures",
        "models": outdir / "models",
        "reports": outdir / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def split_counts(df: pd.DataFrame, split_name: str) -> dict[str, Any]:
    counts = df[TARGET].value_counts().reindex([0, 1], fill_value=0)
    return {
        "split": split_name,
        "n": int(len(df)),
        "class_0": int(counts[0]),
        "class_1": int(counts[1]),
        "diabetes_prevalence": float(df[TARGET].mean()),
    }


def source_split(source: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_dev, test = train_test_split(source, test_size=0.20, stratify=source[TARGET], random_state=seed)
    train, calib = train_test_split(train_dev, test_size=0.25, stratify=train_dev[TARGET], random_state=seed)
    return train.reset_index(drop=True), calib.reset_index(drop=True), test.reset_index(drop=True)


def external_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    calib, test = train_test_split(df, test_size=0.50, stratify=df[TARGET], random_state=seed)
    return calib.reset_index(drop=True), test.reset_index(drop=True)


def calibration_fit_select_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 20 or df[TARGET].nunique() < 2 or df[TARGET].value_counts().min() < 4:
        return df.reset_index(drop=True), df.reset_index(drop=True)
    fit, select = train_test_split(df, test_size=0.50, stratify=df[TARGET], random_state=seed)
    return fit.reset_index(drop=True), select.reset_index(drop=True)


def candidate_estimators(model_name: str, seed: int, quick: bool) -> tuple[list[tuple[str, Any]], bool]:
    n_tree = 120 if quick else 300
    if model_name == "Logistic Regression":
        return [
            ("C=0.25", LogisticRegression(max_iter=5000, C=0.25, class_weight="balanced")),
            ("C=1.0", LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced")),
            ("C=4.0", LogisticRegression(max_iter=5000, C=4.0, class_weight="balanced")),
        ], True
    if model_name == "Random Forest":
        return [
            ("depth=8_leaf=5", RandomForestClassifier(n_estimators=n_tree, max_depth=8, min_samples_leaf=5, random_state=seed, n_jobs=1, class_weight="balanced")),
            ("depth=10_leaf=5", RandomForestClassifier(n_estimators=n_tree, max_depth=10, min_samples_leaf=5, random_state=seed, n_jobs=1, class_weight="balanced")),
            ("depth=none_leaf=10", RandomForestClassifier(n_estimators=n_tree, max_depth=None, min_samples_leaf=10, random_state=seed, n_jobs=1, class_weight="balanced")),
        ], False
    if model_name == "XGBoost":
        import xgboost as xgb
        n_est = 150 if quick else 300
        common = dict(objective="binary:logistic", eval_metric="logloss", random_state=seed, n_jobs=1, tree_method="hist")
        return [
            ("depth=3_lr=0.05", xgb.XGBClassifier(n_estimators=n_est, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, **common)),
            ("depth=5_lr=0.05", xgb.XGBClassifier(n_estimators=n_est, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, **common)),
            ("depth=5_lr=0.03", xgb.XGBClassifier(n_estimators=n_est, learning_rate=0.03, max_depth=5, subsample=0.9, colsample_bytree=0.9, **common)),
        ], False
    if model_name == "LightGBM":
        import lightgbm as lgb
        n_est = 150 if quick else 300
        return [
            ("leaves=15_lr=0.05", lgb.LGBMClassifier(n_estimators=n_est, learning_rate=0.05, num_leaves=15, random_state=seed, n_jobs=1, class_weight="balanced", verbose=-1)),
            ("leaves=31_lr=0.05", lgb.LGBMClassifier(n_estimators=n_est, learning_rate=0.05, num_leaves=31, random_state=seed, n_jobs=1, class_weight="balanced", verbose=-1)),
            ("leaves=31_lr=0.03", lgb.LGBMClassifier(n_estimators=n_est, learning_rate=0.03, num_leaves=31, random_state=seed, n_jobs=1, class_weight="balanced", verbose=-1)),
        ], False
    if model_name == "CatBoost":
        from catboost import CatBoostClassifier
        n_iter = 150 if quick else 300
        common = dict(iterations=n_iter, random_seed=seed, verbose=False, allow_writing_files=False)
        return [
            ("depth=4_lr=0.05", CatBoostClassifier(depth=4, learning_rate=0.05, **common)),
            ("depth=6_lr=0.05", CatBoostClassifier(depth=6, learning_rate=0.05, **common)),
            ("depth=6_lr=0.03", CatBoostClassifier(depth=6, learning_rate=0.03, **common)),
        ], False
    raise ValueError(model_name)


def tune_and_fit_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    quick: bool,
) -> tuple[Pipeline, list[dict[str, Any]]]:
    train_fit, train_val = train_test_split(
        pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True).rename(TARGET)], axis=1),
        test_size=0.20,
        stratify=y_train,
        random_state=seed,
    )
    X_fit = train_fit.drop(columns=[TARGET])
    y_fit = train_fit[TARGET]
    X_val = train_val.drop(columns=[TARGET])
    y_val = train_val[TARGET]
    candidates, scale_numeric = candidate_estimators(model_name, seed, quick)
    rows = []
    fitted_candidates: list[tuple[dict[str, Any], Any, bool]] = []
    for candidate_name, estimator in candidates:
        model = base.make_pipeline(clone(estimator), X_fit, scale_numeric)
        model.fit(X_fit, y_fit)
        p_val = base.positive_probabilities(model, X_val)
        row = {
            "model": model_name,
            "candidate": candidate_name,
            "validation_n": int(len(y_val)),
            "validation_roc_auc": float(roc_auc_score(y_val, p_val)),
            "validation_brier": float(brier_score_loss(y_val, p_val)),
            "validation_ece": base.ece_score(y_val, p_val),
        }
        rows.append(row)
        fitted_candidates.append((row, estimator, scale_numeric))
    best_row, best_estimator, best_scale = sorted(
        fitted_candidates,
        key=lambda item: (-item[0]["validation_roc_auc"], item[0]["validation_brier"], item[0]["validation_ece"]),
    )[0]
    final_model = base.make_pipeline(clone(best_estimator), X_train, best_scale)
    final_model.fit(X_train, y_train)
    for row in rows:
        row["selected"] = row["candidate"] == best_row["candidate"]
    return final_model, rows


def select_probability_calibrator(
    raw_fit: np.ndarray,
    y_fit: pd.Series,
    raw_select: np.ndarray,
    y_select: pd.Series,
    label: dict[str, Any],
) -> tuple[str, base.ProbabilityCalibrator, list[dict[str, Any]]]:
    rows = []
    calibrators = {}
    for method in CALIBRATION_METHODS:
        cal = base.ProbabilityCalibrator(method).fit(raw_fit, y_fit)
        p_select = cal.transform(raw_select)
        rows.append({
            **label,
            "calibration_method": method,
            "selection_n": int(len(y_select)),
            "selection_ece": base.ece_score(y_select, p_select),
            "selection_brier": float(brier_score_loss(y_select, p_select)),
        })
        calibrators[method] = cal
    ranked = pd.DataFrame(rows)
    ranked["rank_score"] = ranked["selection_ece"].rank() + ranked["selection_brier"].rank()
    best_method = str(ranked.sort_values(["rank_score", "selection_ece", "selection_brier"]).iloc[0]["calibration_method"])
    for row in rows:
        row["selected"] = row["calibration_method"] == best_method
    return best_method, calibrators[best_method], rows


def label_shift_adjust(p: np.ndarray, source_prev: float, target_prev: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    r1 = target_prev / max(source_prev, 1e-6)
    r0 = (1 - target_prev) / max(1 - source_prev, 1e-6)
    adjusted = (r1 * p) / (r1 * p + r0 * (1 - p))
    return np.clip(adjusted, 1e-6, 1 - 1e-6)


def positive_from_probs(p: np.ndarray) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return np.clip(arr[:, 1], 1e-6, 1 - 1e-6)
    return np.clip(arr, 1e-6, 1 - 1e-6)


def as_probability_matrix(p: np.ndarray) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return np.clip(arr, 1e-6, 1 - 1e-6)
    return base.probability_matrix(arr)


def mean_psi_for_features(psi_table: pd.DataFrame, target_name: str, features: list[str]) -> float:
    values = psi_table.loc[
        psi_table["target_dataset"].eq(target_name) & psi_table["feature"].isin(features),
        "psi",
    ].dropna()
    return float(values.mean()) if len(values) else float("nan")


def compute_mean_psi(source_df: pd.DataFrame, target_df: pd.DataFrame, features: list[str]) -> float:
    values = []
    for feature in features:
        if feature not in source_df or feature not in target_df:
            continue
        if not pd.api.types.is_numeric_dtype(source_df[feature]) and not pd.api.types.is_numeric_dtype(target_df[feature]):
            continue
        value = base.psi(source_df[feature], target_df[feature])
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def sampled_indices(n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    if size <= 0:
        return np.asarray([], dtype=int)
    return rng.choice(n, size=size, replace=size > n)


def domain_shift_auc(source_df: pd.DataFrame, target_df: pd.DataFrame, features: list[str], seed: int) -> float:
    source_x = base.prepare_X(source_df, features).assign(domain=0)
    target_x = base.prepare_X(target_df, features).assign(domain=1)
    data = pd.concat([source_x, target_x], ignore_index=True)
    y = data.pop("domain")
    if y.nunique() < 2:
        return float("nan")
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        y,
        test_size=0.30,
        stratify=y,
        random_state=seed,
    )
    model = Pipeline([
        ("preprocess", base.build_preprocessor(X_train, scale_numeric=True)),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
    ])
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, p))


def shift_aware_source_weight(
    mean_psi: float,
    domain_auc: float,
    target_n: int,
    prevalence_gap: float,
    alpha: float = 2.0,
    beta: float = 3.0,
    gamma: float = 2.0,
) -> float:
    psi_value = 0.0 if not np.isfinite(mean_psi) else max(mean_psi, 0.0)
    domain_gap = 0.0 if not np.isfinite(domain_auc) else max(domain_auc - 0.5, 0.0)
    n_factor = 1.0 / np.sqrt(max(target_n, 1))
    prevalence_factor = np.exp(-gamma * abs(prevalence_gap))
    weight = np.exp(-alpha * psi_value) * np.exp(-beta * domain_gap) * n_factor * prevalence_factor
    return float(np.clip(weight, 0.0, 0.20))


def referral_metrics(
    y_true: pd.Series | np.ndarray,
    sets: np.ndarray,
    probs: np.ndarray | None = None,
) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=int)
    sizes = sets.sum(axis=1)
    referral = sizes != 1
    referred_positive = labels[referral] if referral.any() else np.asarray([])
    metrics = {
        "single_label_rate": float((sizes == 1).mean()),
        "referral_rate": float(referral.mean()),
        "uncertain_set_rate": float((sizes == 2).mean()),
        "diabetes_prevalence_in_referred": float(referred_positive.mean()) if len(referred_positive) else np.nan,
    }
    if probs is not None:
        prob_matrix = as_probability_matrix(probs)
        confidence = prob_matrix.max(axis=1)
        non_referred = sizes == 1
        single_predictions = np.argmax(sets, axis=1)
        metrics.update({
            "mean_confidence_referred": float(confidence[referral].mean()) if referral.any() else np.nan,
            "mean_confidence_non_referred": float(confidence[non_referred].mean()) if non_referred.any() else np.nan,
            "non_referred_error_rate": float((single_predictions[non_referred] != labels[non_referred]).mean()) if non_referred.any() else np.nan,
        })
    return metrics


def clinical_utility_score(
    y_true: pd.Series | np.ndarray,
    sets: np.ndarray,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
    uncertain_cost: float = 2.0,
    empty_cost: float = 3.0,
) -> float:
    labels = np.asarray(y_true, dtype=int)
    total_cost = 0.0
    for label, pred_set in zip(labels, sets):
        included = np.flatnonzero(pred_set)
        if len(included) == 2:
            total_cost += uncertain_cost
        elif len(included) == 0:
            total_cost += empty_cost
        elif included[0] == 0 and label == 1:
            total_cost += fn_cost
        elif included[0] == 1 and label == 0:
            total_cost += fp_cost
    return float(total_cost / max(len(labels), 1))


def extended_conformal_metrics(
    y_true: pd.Series,
    sets: np.ndarray,
    label: dict[str, Any],
    probs: np.ndarray | None = None,
    bootstrap_repeats: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    row = base.conformal_metrics(y_true, sets, label)
    class_0 = row.get("class_0_conditional_coverage", np.nan)
    class_1 = row.get("class_1_conditional_coverage", np.nan)
    row["class_coverage_gap_abs"] = float(abs(class_0 - class_1)) if np.isfinite(class_0) and np.isfinite(class_1) else np.nan
    row["clinical_cost"] = clinical_utility_score(y_true, sets)
    row.update(referral_metrics(y_true, sets, probs))
    if bootstrap_repeats > 0 and abs(float(label.get("confidence_level", PRIMARY_CONFIDENCE)) - PRIMARY_CONFIDENCE) < 1e-12:
        ci = bootstrap_coverage_ci(y_true, sets, n_boot=bootstrap_repeats, seed=seed)
        row.update(ci)
    return row


def bootstrap_coverage_ci(
    y_true: pd.Series | np.ndarray,
    sets: np.ndarray,
    n_boot: int = 300,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(y_true, dtype=int)
    covered = sets[np.arange(len(labels)), labels]
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), size=len(labels))
        values.append(float(covered[idx].mean()))
    return {
        "coverage_bootstrap_mean": float(np.mean(values)),
        "coverage_ci_low": float(np.percentile(values, 2.5)),
        "coverage_ci_high": float(np.percentile(values, 97.5)),
    }


def weighted_quantile(scores: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(scores)
    s = np.asarray(scores)[order]
    w = np.asarray(weights, dtype=float)[order]
    w = np.clip(w, 1e-8, None)
    cdf = np.cumsum(w) / np.sum(w)
    return float(s[np.searchsorted(cdf, q, side="left").clip(0, len(s) - 1)])


def weighted_atcp_quantile(
    p_source_calib: np.ndarray,
    y_source_calib: pd.Series,
    p_target_calib: np.ndarray,
    y_target_calib: pd.Series,
    confidence: float,
    source_weight: float,
    target_weight: float = 1.0,
) -> float:
    source_probs = as_probability_matrix(p_source_calib)
    target_probs = as_probability_matrix(p_target_calib)
    labels_source = y_source_calib.to_numpy(dtype=int)
    labels_target = y_target_calib.to_numpy(dtype=int)
    scores_source = 1 - source_probs[np.arange(len(labels_source)), labels_source]
    scores_target = 1 - target_probs[np.arange(len(labels_target)), labels_target]
    scores = np.concatenate([scores_source, scores_target])
    weights = np.concatenate([
        np.full(len(scores_source), source_weight, dtype=float),
        np.full(len(scores_target), target_weight, dtype=float),
    ])
    return weighted_quantile(scores, weights, confidence)


def weighted_class_conditional_quantiles(
    p_source_calib: np.ndarray,
    y_source_calib: pd.Series,
    p_target_calib: np.ndarray,
    y_target_calib: pd.Series,
    confidence: float,
    source_weight: float,
    target_weight: float = 1.0,
) -> dict[int, float]:
    source_probs = as_probability_matrix(p_source_calib)
    target_probs = as_probability_matrix(p_target_calib)
    labels_source = y_source_calib.to_numpy(dtype=int)
    labels_target = y_target_calib.to_numpy(dtype=int)
    scores_source = 1 - source_probs[np.arange(len(labels_source)), labels_source]
    scores_target = 1 - target_probs[np.arange(len(labels_target)), labels_target]
    labels = np.concatenate([labels_source, labels_target])
    scores = np.concatenate([scores_source, scores_target])
    weights = np.concatenate([
        np.full(len(scores_source), source_weight, dtype=float),
        np.full(len(scores_target), target_weight, dtype=float),
    ])
    q_by_class: dict[int, float] = {}
    for cls in (0, 1):
        mask = labels == cls
        q_by_class[cls] = weighted_quantile(scores[mask], weights[mask], confidence) if mask.any() else float("inf")
    return q_by_class


def risk_asymmetric_confidences(confidence: float, delta: float = RISK_ASYMMETRY_DELTA) -> dict[int, float]:
    return {
        0: float(np.clip(confidence - delta, 0.50, 0.99)),
        1: float(np.clip(confidence + delta, 0.50, 0.99)),
    }


def weighted_risk_asymmetric_quantiles(
    p_source_calib: np.ndarray,
    y_source_calib: pd.Series,
    p_target_calib: np.ndarray,
    y_target_calib: pd.Series,
    confidence: float,
    source_weight: float,
    target_weight: float = 1.0,
) -> dict[int, float]:
    class_confidence = risk_asymmetric_confidences(confidence)
    source_probs = as_probability_matrix(p_source_calib)
    target_probs = as_probability_matrix(p_target_calib)
    labels_source = y_source_calib.to_numpy(dtype=int)
    labels_target = y_target_calib.to_numpy(dtype=int)
    scores_source = 1 - source_probs[np.arange(len(labels_source)), labels_source]
    scores_target = 1 - target_probs[np.arange(len(labels_target)), labels_target]
    labels = np.concatenate([labels_source, labels_target])
    scores = np.concatenate([scores_source, scores_target])
    weights = np.concatenate([
        np.full(len(scores_source), source_weight, dtype=float),
        np.full(len(scores_target), target_weight, dtype=float),
    ])
    q_by_class: dict[int, float] = {}
    for cls in (0, 1):
        mask = labels == cls
        q_by_class[cls] = weighted_quantile(scores[mask], weights[mask], class_confidence[cls]) if mask.any() else float("inf")
    return q_by_class


def weighted_class_specific_quantiles(
    p_source_calib: np.ndarray,
    y_source_calib: pd.Series,
    p_target_calib: np.ndarray,
    y_target_calib: pd.Series,
    class_confidences: dict[int, float],
    source_weight: float,
    target_weight: float = 1.0,
) -> dict[int, float]:
    source_probs = as_probability_matrix(p_source_calib)
    target_probs = as_probability_matrix(p_target_calib)
    labels_source = y_source_calib.to_numpy(dtype=int)
    labels_target = y_target_calib.to_numpy(dtype=int)
    scores_source = 1 - source_probs[np.arange(len(labels_source)), labels_source]
    scores_target = 1 - target_probs[np.arange(len(labels_target)), labels_target]
    labels = np.concatenate([labels_source, labels_target])
    scores = np.concatenate([scores_source, scores_target])
    weights = np.concatenate([
        np.full(len(scores_source), source_weight, dtype=float),
        np.full(len(scores_target), target_weight, dtype=float),
    ])
    q_by_class: dict[int, float] = {}
    for cls in (0, 1):
        mask = labels == cls
        q_by_class[cls] = weighted_quantile(scores[mask], weights[mask], class_confidences[cls]) if mask.any() else float("inf")
    return q_by_class


def class_conditional_sets(probs: np.ndarray, q_by_class: dict[int, float]) -> np.ndarray:
    prob_matrix = as_probability_matrix(probs)
    sets = np.zeros_like(prob_matrix, dtype=bool)
    for cls in (0, 1):
        sets[:, cls] = (1 - prob_matrix[:, cls]) <= q_by_class.get(cls, float("inf"))
    return sets


def diabetes_protected_score(metrics: dict[str, Any], confidence: float) -> float:
    diabetes_coverage = metrics.get("class_1_conditional_coverage", np.nan)
    diabetes_undercoverage = max(0.0, confidence - diabetes_coverage) if np.isfinite(diabetes_coverage) else 1.0
    return float(
        5.0 * abs(metrics["marginal_coverage"] - confidence)
        + 2.0 * diabetes_undercoverage
        + 1.5 * metrics.get("class_coverage_gap_abs", 0.0)
        + 0.8 * metrics["average_prediction_set_size"]
        + 0.5 * metrics.get("clinical_cost", 0.0)
    )


def adjusted_dp_confidence_grid(confidence: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    offset = confidence - PRIMARY_CONFIDENCE
    class_0 = tuple(float(np.clip(v + offset, 0.50, 0.99)) for v in DP_CLASS0_CONFIDENCE_GRID)
    class_1 = tuple(float(np.clip(v + offset, 0.50, 0.99)) for v in DP_CLASS1_CONFIDENCE_GRID)
    return class_0, class_1


def select_diabetes_protected_params(
    y_source_calib: pd.Series,
    p_source_calib: np.ndarray,
    y_target_fit: pd.Series,
    p_target_fit: np.ndarray,
    y_target_select: pd.Series,
    p_target_select: np.ndarray,
    confidence: float,
    source_weight: float,
    seed: int,
) -> dict[str, Any]:
    select_probs = as_probability_matrix(p_target_select)
    class_0_grid, class_1_grid = adjusted_dp_confidence_grid(confidence)
    candidates = []
    for class_0_conf in class_0_grid:
        for class_1_conf in class_1_grid:
            if class_1_conf < class_0_conf:
                continue
            class_confidences = {0: class_0_conf, 1: class_1_conf}
            q_by_class = weighted_class_specific_quantiles(
                p_source_calib,
                y_source_calib,
                p_target_fit,
                y_target_fit,
                class_confidences,
                source_weight=source_weight,
            )
            sets = class_conditional_sets(select_probs, q_by_class)
            metrics = extended_conformal_metrics(
                y_target_select,
                sets,
                {"confidence_level": confidence},
                probs=select_probs,
                bootstrap_repeats=0,
                seed=seed,
            )
            candidates.append({
                "class_0_target_confidence": class_0_conf,
                "class_1_target_confidence": class_1_conf,
                "score": diabetes_protected_score(metrics, confidence),
                "validation_coverage": metrics["marginal_coverage"],
                "validation_set_size": metrics["average_prediction_set_size"],
                "validation_class_0_coverage": metrics["class_0_conditional_coverage"],
                "validation_class_1_coverage": metrics["class_1_conditional_coverage"],
                "validation_class_gap": metrics["class_coverage_gap_abs"],
                "validation_clinical_cost": metrics["clinical_cost"],
                "validation_referral_rate": metrics["referral_rate"],
            })
    return sorted(candidates, key=lambda row: row["score"])[0]


def multi_objective_score(metrics: dict[str, Any], ece: float, confidence: float) -> float:
    undercoverage = max(0.0, confidence - metrics["marginal_coverage"])
    return float(
        8.0 * undercoverage
        + 2.0 * abs(metrics["marginal_coverage"] - confidence)
        + 0.75 * metrics["average_prediction_set_size"]
        + 2.0 * ece
        + 2.0 * metrics.get("class_coverage_gap_abs", 0.0)
        + 0.50 * metrics.get("clinical_cost", 0.0)
    )


def select_multi_objective_weight(
    y_source_calib: pd.Series,
    p_source_calib: np.ndarray,
    y_target_fit: pd.Series,
    p_target_fit: np.ndarray,
    y_target_select: pd.Series,
    p_target_select: np.ndarray,
    confidence: float,
    shift_source_weight: float,
    seed: int,
) -> dict[str, Any]:
    select_probs = as_probability_matrix(p_target_select)
    candidates: list[dict[str, Any]] = []
    candidate_weights = sorted(set(ATCP_SOURCE_WEIGHT_GRID + (float(shift_source_weight),)))
    for source_weight in candidate_weights:
        q_hat = weighted_atcp_quantile(
            p_source_calib,
            y_source_calib,
            p_target_fit,
            y_target_fit,
            confidence,
            source_weight=source_weight,
        )
        sets = base.conformal_sets(select_probs, q_hat)
        metrics = extended_conformal_metrics(
            y_target_select,
            sets,
            {"confidence_level": confidence},
            bootstrap_repeats=0,
            seed=seed,
        )
        ece = base.ece_score(y_target_select, positive_from_probs(p_target_select))
        candidates.append({
            "candidate_type": "weighted",
            "source_weight": float(source_weight),
            "score": multi_objective_score(metrics, ece, confidence),
            "validation_coverage": metrics["marginal_coverage"],
            "validation_set_size": metrics["average_prediction_set_size"],
            "validation_class_gap": metrics["class_coverage_gap_abs"],
            "validation_clinical_cost": metrics["clinical_cost"],
        })

    q_target = base.conformal_quantile(as_probability_matrix(p_target_fit), y_target_fit, confidence)
    target_sets = base.conformal_sets(select_probs, q_target)
    target_metrics = extended_conformal_metrics(
        y_target_select,
        target_sets,
        {"confidence_level": confidence},
        bootstrap_repeats=0,
        seed=seed,
    )
    target_ece = base.ece_score(y_target_select, positive_from_probs(p_target_select))
    candidates.append({
        "candidate_type": "target_only",
        "source_weight": 0.0,
        "score": multi_objective_score(target_metrics, target_ece, confidence),
        "validation_coverage": target_metrics["marginal_coverage"],
        "validation_set_size": target_metrics["average_prediction_set_size"],
        "validation_class_gap": target_metrics["class_coverage_gap_abs"],
        "validation_clinical_cost": target_metrics["clinical_cost"],
    })
    return sorted(candidates, key=lambda row: row["score"])[0]


def select_atcp_source_weight(
    y_source_calib: pd.Series,
    p_source_calib: np.ndarray,
    y_target_fit: pd.Series,
    p_target_fit: np.ndarray,
    y_target_select: pd.Series,
    p_target_select: np.ndarray,
    confidence: float,
) -> tuple[float, float, float]:
    rows = []
    select_probs = as_probability_matrix(p_target_select)
    for source_weight in ATCP_SOURCE_WEIGHT_GRID:
        q_hat = weighted_atcp_quantile(
            p_source_calib,
            y_source_calib,
            p_target_fit,
            y_target_fit,
            confidence,
            source_weight=source_weight,
        )
        sets = base.conformal_sets(select_probs, q_hat)
        metrics = base.conformal_metrics(
            y_target_select,
            sets,
            {"confidence_level": confidence, "source_weight": source_weight},
        )
        rows.append({
            "source_weight": source_weight,
            "validation_coverage": metrics["marginal_coverage"],
            "validation_set_size": metrics["average_prediction_set_size"],
            "undercoverage": max(0.0, confidence - metrics["marginal_coverage"]),
            "coverage_gap_abs": abs(metrics["marginal_coverage"] - confidence),
        })
    ranked = sorted(rows, key=lambda r: (r["undercoverage"] > 0, r["coverage_gap_abs"], r["validation_set_size"]))
    best = ranked[0]
    return float(best["source_weight"]), float(best["validation_coverage"]), float(best["validation_set_size"])


def estimate_density_ratio_weights(
    source_calib: pd.DataFrame,
    target_calib: pd.DataFrame,
    features: list[str],
    seed: int,
) -> np.ndarray:
    source_x = base.prepare_X(source_calib, features).assign(domain=0)
    target_x = base.prepare_X(target_calib, features).assign(domain=1)
    combined = pd.concat([source_x, target_x], ignore_index=True)
    y_domain = combined.pop("domain")
    domain_model = Pipeline([
        ("preprocess", base.build_preprocessor(combined, scale_numeric=True)),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
    ])
    domain_model.fit(combined, y_domain)
    p_t = np.clip(domain_model.predict_proba(base.prepare_X(source_calib, features))[:, 1], 1e-4, 1 - 1e-4)
    ratio = (p_t / (1 - p_t)) * (len(source_calib) / max(len(target_calib), 1))
    return np.clip(ratio, 0.05, 20.0)


def conformal_rows_for_scenario(
    y_test: pd.Series,
    p_test: np.ndarray,
    y_source_calib: pd.Series,
    p_source_calib: np.ndarray,
    y_target_calib: pd.Series,
    p_target_calib: np.ndarray,
    y_target_fit: pd.Series,
    p_target_fit: np.ndarray,
    y_target_select: pd.Series,
    p_target_select: np.ndarray,
    source_weights: np.ndarray,
    shift_source_weight: float,
    shift_label: dict[str, Any],
    label: dict[str, Any],
    bootstrap_repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    test_probs = as_probability_matrix(p_test)
    source_probs = as_probability_matrix(p_source_calib)
    labels_source = y_source_calib.to_numpy(dtype=int)
    scores_source = 1 - source_probs[np.arange(len(labels_source)), labels_source]
    labels_target = y_target_calib.to_numpy(dtype=int)
    target_probs = as_probability_matrix(p_target_calib)
    for confidence in EVAL_CONFIDENCE_LEVELS:
        q_source = base.conformal_quantile(source_probs, y_source_calib, confidence)
        sets_source = base.conformal_sets(test_probs, q_source)
        rows.append(extended_conformal_metrics(y_test, sets_source, {
            **label, "confidence_level": confidence, "conformal_method": "source_split_conformal",
        }, probs=test_probs, bootstrap_repeats=bootstrap_repeats, seed=seed))

        q_weighted = weighted_quantile(scores_source, source_weights, confidence)
        sets_weighted = base.conformal_sets(test_probs, q_weighted)
        rows.append(extended_conformal_metrics(y_test, sets_weighted, {
            **label, "confidence_level": confidence, "conformal_method": "importance_weighted_source_conformal",
        }, probs=test_probs, bootstrap_repeats=bootstrap_repeats, seed=seed))

        q_target = base.conformal_quantile(target_probs, y_target_calib, confidence)
        sets_target = base.conformal_sets(test_probs, q_target)
        rows.append(extended_conformal_metrics(y_test, sets_target, {
            **label, "confidence_level": confidence, "conformal_method": "target_only_conformal",
        }, probs=test_probs, bootstrap_repeats=bootstrap_repeats, seed=seed))

        q_atcp = weighted_atcp_quantile(
            p_source_calib,
            y_source_calib,
            p_target_calib,
            y_target_calib,
            confidence,
            source_weight=1.0,
        )
        sets_atcp = base.conformal_sets(test_probs, q_atcp)
        rows.append(extended_conformal_metrics(y_test, sets_atcp, {
            **label,
            "confidence_level": confidence,
            "conformal_method": "adaptive_transport_conformal",
            "selected_source_weight": 1.0,
            "source_weight_strategy": "unweighted",
            "gate_validation_coverage": np.nan,
            "fallback_used": False,
            **shift_label,
        }, probs=test_probs, bootstrap_repeats=bootstrap_repeats, seed=seed))

        selected_weight, gate_coverage, gate_set_size = select_atcp_source_weight(
            y_source_calib,
            p_source_calib,
            y_target_fit,
            p_target_fit,
            y_target_select,
            p_target_select,
            confidence,
        )
        q_weighted_atcp = weighted_atcp_quantile(
            p_source_calib,
            y_source_calib,
            p_target_calib,
            y_target_calib,
            confidence,
            source_weight=selected_weight,
        )
        sets_weighted_atcp = base.conformal_sets(test_probs, q_weighted_atcp)
        rows.append(extended_conformal_metrics(y_test, sets_weighted_atcp, {
            **label,
            "confidence_level": confidence,
            "conformal_method": "weighted_target_dominant_atcp",
            "selected_source_weight": selected_weight,
            "source_weight_strategy": "grid_selected",
            "gate_validation_coverage": gate_coverage,
            "gate_validation_set_size": gate_set_size,
            "fallback_used": False,
            **shift_label,
        }, probs=test_probs, bootstrap_repeats=bootstrap_repeats, seed=seed))

        q_shift_atcp = weighted_atcp_quantile(
            p_source_calib,
            y_source_calib,
            p_target_calib,
            y_target_calib,
            confidence,
            source_weight=shift_source_weight,
        )
        sets_shift_atcp = base.conformal_sets(test_probs, q_shift_atcp)
        rows.append(extended_conformal_metrics(y_test, sets_shift_atcp, {
            **label,
            "confidence_level": confidence,
            "conformal_method": "shift_aware_source_trust_atcp",
            "selected_source_weight": shift_source_weight,
            "source_weight_strategy": "shift_formula",
            "gate_validation_coverage": np.nan,
            "gate_validation_set_size": np.nan,
            "fallback_used": False,
            **shift_label,
        }, probs=test_probs, bootstrap_repeats=bootstrap_repeats, seed=seed))

        q_by_class = weighted_class_conditional_quantiles(
            p_source_calib,
            y_source_calib,
            p_target_calib,
            y_target_calib,
            confidence,
            source_weight=selected_weight,
        )
        sets_class_conditional = class_conditional_sets(test_probs, q_by_class)
        rows.append(extended_conformal_metrics(y_test, sets_class_conditional, {
            **label,
            "confidence_level": confidence,
            "conformal_method": "class_conditional_weighted_atcp",
            "selected_source_weight": selected_weight,
            "source_weight_strategy": "grid_selected",
            "gate_validation_coverage": gate_coverage,
            "gate_validation_set_size": gate_set_size,
            "fallback_used": False,
            **shift_label,
        }, probs=test_probs, bootstrap_repeats=bootstrap_repeats, seed=seed))

        q_risk_shift = weighted_risk_asymmetric_quantiles(
            p_source_calib,
            y_source_calib,
            p_target_calib,
            y_target_calib,
            confidence,
            source_weight=shift_source_weight,
        )
        sets_risk_shift = class_conditional_sets(test_probs, q_risk_shift)
        risk_conf = risk_asymmetric_confidences(confidence)
        rows.append(extended_conformal_metrics(y_test, sets_risk_shift, {
            **label,
            "confidence_level": confidence,
            "class_0_target_confidence": risk_conf[0],
            "class_1_target_confidence": risk_conf[1],
            "conformal_method": "risk_asymmetric_shift_atcp",
            "selected_source_weight": shift_source_weight,
            "source_weight_strategy": "shift_formula",
            "gate_validation_coverage": np.nan,
            "gate_validation_set_size": np.nan,
            "fallback_used": False,
            **shift_label,
        }, probs=test_probs, bootstrap_repeats=bootstrap_repeats, seed=seed))

        dp_selection = select_diabetes_protected_params(
            y_source_calib,
            p_source_calib,
            y_target_fit,
            p_target_fit,
            y_target_select,
            p_target_select,
            confidence,
            shift_source_weight,
            seed,
        )
        dp_class_confidences = {
            0: dp_selection["class_0_target_confidence"],
            1: dp_selection["class_1_target_confidence"],
        }
        q_dp = weighted_class_specific_quantiles(
            p_source_calib,
            y_source_calib,
            p_target_calib,
            y_target_calib,
            dp_class_confidences,
            source_weight=shift_source_weight,
        )
        sets_dp = class_conditional_sets(test_probs, q_dp)
        rows.append(extended_conformal_metrics(y_test, sets_dp, {
            **label,
            "confidence_level": confidence,
            "class_0_target_confidence": dp_class_confidences[0],
            "class_1_target_confidence": dp_class_confidences[1],
            "conformal_method": "dp_shift_atcp",
            "selected_source_weight": shift_source_weight,
            "source_weight_strategy": "shift_formula",
            "gate_validation_coverage": dp_selection["validation_coverage"],
            "gate_validation_set_size": dp_selection["validation_set_size"],
            "gate_validation_class_0_coverage": dp_selection["validation_class_0_coverage"],
            "gate_validation_class_1_coverage": dp_selection["validation_class_1_coverage"],
            "gate_validation_class_gap": dp_selection["validation_class_gap"],
            "gate_validation_clinical_cost": dp_selection["validation_clinical_cost"],
            "gate_validation_referral_rate": dp_selection["validation_referral_rate"],
            "multi_objective_score": dp_selection["score"],
            "fallback_used": False,
            **shift_label,
        }, probs=test_probs, bootstrap_repeats=bootstrap_repeats, seed=seed))

        if gate_coverage < confidence:
            q_safety = q_target
            safety_method = "target_only_fallback"
            fallback_used = True
        else:
            q_safety = q_weighted_atcp
            safety_method = "weighted_atcp_accepted"
            fallback_used = False
        sets_safety = base.conformal_sets(test_probs, q_safety)
        rows.append(extended_conformal_metrics(y_test, sets_safety, {
            **label,
            "confidence_level": confidence,
            "conformal_method": "safety_gated_atcp",
            "selected_source_weight": selected_weight,
            "source_weight_strategy": "grid_selected",
            "gate_validation_coverage": gate_coverage,
            "gate_validation_set_size": gate_set_size,
            "fallback_used": fallback_used,
            "safety_decision": safety_method,
            **shift_label,
        }, probs=test_probs, bootstrap_repeats=bootstrap_repeats, seed=seed))

        shift_select_q = weighted_atcp_quantile(
            p_source_calib,
            y_source_calib,
            p_target_fit,
            y_target_fit,
            confidence,
            source_weight=shift_source_weight,
        )
        shift_select_sets = base.conformal_sets(as_probability_matrix(p_target_select), shift_select_q)
        shift_gate_metrics = extended_conformal_metrics(
            y_target_select,
            shift_select_sets,
            {"confidence_level": confidence},
            bootstrap_repeats=0,
            seed=seed,
        )
        if shift_gate_metrics["marginal_coverage"] < confidence:
            q_shift_safety = q_target
            shift_safety_method = "target_only_fallback"
            shift_fallback = True
        else:
            q_shift_safety = q_shift_atcp
            shift_safety_method = "shift_atcp_accepted"
            shift_fallback = False
        sets_shift_safety = base.conformal_sets(test_probs, q_shift_safety)
        rows.append(extended_conformal_metrics(y_test, sets_shift_safety, {
            **label,
            "confidence_level": confidence,
            "conformal_method": "safety_gated_shift_atcp",
            "selected_source_weight": shift_source_weight,
            "source_weight_strategy": "shift_formula",
            "gate_validation_coverage": shift_gate_metrics["marginal_coverage"],
            "gate_validation_set_size": shift_gate_metrics["average_prediction_set_size"],
            "fallback_used": shift_fallback,
            "safety_decision": shift_safety_method,
            **shift_label,
        }, probs=test_probs, bootstrap_repeats=bootstrap_repeats, seed=seed))

        multi = select_multi_objective_weight(
            y_source_calib,
            p_source_calib,
            y_target_fit,
            p_target_fit,
            y_target_select,
            p_target_select,
            confidence,
            shift_source_weight,
            seed,
        )
        if multi["candidate_type"] == "target_only" or multi["validation_coverage"] < confidence:
            q_multi = q_target
            multi_decision = "target_only_fallback"
            multi_fallback = True
        else:
            q_multi = weighted_atcp_quantile(
                p_source_calib,
                y_source_calib,
                p_target_calib,
                y_target_calib,
                confidence,
                source_weight=multi["source_weight"],
            )
            multi_decision = "multi_objective_weighted_atcp"
            multi_fallback = False
        sets_multi = base.conformal_sets(test_probs, q_multi)
        rows.append(extended_conformal_metrics(y_test, sets_multi, {
            **label,
            "confidence_level": confidence,
            "conformal_method": "multi_objective_safety_gated_atcp",
            "selected_source_weight": multi["source_weight"],
            "source_weight_strategy": "multi_objective",
            "gate_validation_coverage": multi["validation_coverage"],
            "gate_validation_set_size": multi["validation_set_size"],
            "gate_validation_class_gap": multi["validation_class_gap"],
            "gate_validation_clinical_cost": multi["validation_clinical_cost"],
            "multi_objective_score": multi["score"],
            "fallback_used": multi_fallback,
            "safety_decision": multi_decision,
            **shift_label,
        }, probs=test_probs, bootstrap_repeats=bootstrap_repeats, seed=seed))

        if dp_selection["validation_coverage"] < confidence:
            sets_dp_multi = sets_target
            dp_multi_decision = "target_only_fallback"
            dp_multi_fallback = True
        else:
            sets_dp_multi = sets_dp
            dp_multi_decision = "diabetes_protected_shift_atcp"
            dp_multi_fallback = False
        rows.append(extended_conformal_metrics(y_test, sets_dp_multi, {
            **label,
            "confidence_level": confidence,
            "class_0_target_confidence": dp_class_confidences[0],
            "class_1_target_confidence": dp_class_confidences[1],
            "conformal_method": "multi_objective_dp_shift_atcp",
            "selected_source_weight": shift_source_weight,
            "source_weight_strategy": "shift_formula",
            "gate_validation_coverage": dp_selection["validation_coverage"],
            "gate_validation_set_size": dp_selection["validation_set_size"],
            "gate_validation_class_0_coverage": dp_selection["validation_class_0_coverage"],
            "gate_validation_class_1_coverage": dp_selection["validation_class_1_coverage"],
            "gate_validation_class_gap": dp_selection["validation_class_gap"],
            "gate_validation_clinical_cost": dp_selection["validation_clinical_cost"],
            "gate_validation_referral_rate": dp_selection["validation_referral_rate"],
            "multi_objective_score": dp_selection["score"],
            "fallback_used": dp_multi_fallback,
            "safety_decision": dp_multi_decision,
            **shift_label,
        }, probs=test_probs, bootstrap_repeats=bootstrap_repeats, seed=seed))
    return rows


def stratified_subsample(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if size >= len(df):
        return df.reset_index(drop=True)
    _, sample = train_test_split(df, test_size=size, stratify=df[TARGET], random_state=seed)
    return sample.reset_index(drop=True)


def plot_ablation(ablation: pd.DataFrame, paths: dict[str, Path], dpi: int) -> None:
    primary = ablation.loc[ablation["confidence_level"].eq(PRIMARY_CONFIDENCE)].copy()
    if primary.empty:
        return
    sns.set_theme(style="whitegrid", context="talk")
    method_labels = {
        "adaptive_transport_conformal": "ATCP",
        "weighted_target_dominant_atcp": "Weighted ATCP",
        "shift_aware_source_trust_atcp": "SHIFT-ATCP",
        "class_conditional_weighted_atcp": "Class-conditional ATCP",
        "risk_asymmetric_shift_atcp": "Risk-asymmetric SHIFT",
        "dp_shift_atcp": "DP-SHIFT",
        "safety_gated_atcp": "Safety-gated ATCP",
        "safety_gated_shift_atcp": "Safety-gated SHIFT",
        "multi_objective_safety_gated_atcp": "Multi-objective SHIFT",
        "multi_objective_dp_shift_atcp": "Multi-objective DP-SHIFT",
    }
    primary["method_label"] = primary["conformal_method"].map(method_labels).fillna(primary["conformal_method"])
    for metric, ylabel, filename in [
        ("marginal_coverage", "90% ATCP Coverage", "figure_target_calibration_size_ablation_coverage.png"),
        ("ece", "ECE", "figure_target_calibration_size_ablation_ece.png"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
        for ax, target in zip(axes, ["NHANES", "Pima"]):
            data = primary.loc[primary["target_dataset"].eq(target)]
            if data.empty:
                continue
            for method_label, method_data in data.groupby("method_label"):
                summary = method_data.groupby("calibration_size", as_index=False)[metric].agg(["mean", "std"]).reset_index()
                x_values = summary["calibration_size"].to_numpy(dtype=float)
                y_values = summary["mean"].to_numpy(dtype=float)
                y_std = summary["std"].fillna(0).to_numpy(dtype=float)
                ax.plot(x_values, y_values, marker="o", linewidth=2.5, label=method_label)
                ax.fill_between(x_values, y_values - y_std, y_values + y_std, alpha=0.12)
            if metric == "marginal_coverage":
                ax.axhline(PRIMARY_CONFIDENCE, color="black", linestyle="--", linewidth=2, label="Nominal 90%")
            ax.set_title(target)
            ax.set_xlabel("Target calibration size")
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="both", labelsize=14)
            ax.legend(loc="best", fontsize=10, frameon=True)
        fig.tight_layout()
        fig.savefig(paths["figures"] / filename, dpi=dpi)
        plt.close(fig)


def plot_shift_aware(conformal: pd.DataFrame, paths: dict[str, Path], dpi: int) -> None:
    data = conformal.loc[
        conformal["confidence_level"].eq(PRIMARY_CONFIDENCE)
        & conformal["scenario"].isin(["naive_transfer", "adaptive_calibration_target_selected", "label_shift_adjusted"])
    ].copy()
    if data.empty:
        return
    data["method_label"] = data["scenario"].str.replace("_", " ").str.title() + "\n" + data["conformal_method"].str.replace("_", " ").str.title()
    summary = data.groupby(["target_dataset", "conformal_method"], as_index=False).agg(
        coverage=("marginal_coverage", "mean"),
        set_size=("average_prediction_set_size", "mean"),
    )
    labels = {
        "source_split_conformal": "Source split",
        "importance_weighted_source_conformal": "Importance-weighted source",
        "adaptive_transport_conformal": "ATCP",
        "weighted_target_dominant_atcp": "Weighted ATCP",
        "shift_aware_source_trust_atcp": "SHIFT-ATCP",
        "class_conditional_weighted_atcp": "Class-conditional ATCP",
        "risk_asymmetric_shift_atcp": "Risk-asymmetric SHIFT",
        "dp_shift_atcp": "DP-SHIFT",
        "safety_gated_atcp": "Safety-gated ATCP",
        "safety_gated_shift_atcp": "Safety-gated SHIFT",
        "multi_objective_safety_gated_atcp": "Multi-objective SHIFT",
        "multi_objective_dp_shift_atcp": "Multi-objective DP-SHIFT",
        "target_only_conformal": "Target-only",
    }
    summary["method_label"] = summary["conformal_method"].map(labels).fillna(summary["conformal_method"])
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    for ax, target in zip(axes, ["NHANES", "Pima"]):
        sub = summary.loc[summary["target_dataset"].eq(target)].sort_values("coverage")
        sns.barplot(data=sub, y="method_label", x="coverage", ax=ax, color="#4C78A8")
        ax.axvline(PRIMARY_CONFIDENCE, color="black", linestyle="--", linewidth=2)
        ax.set_xlim(0, 1.0)
        ax.set_title(target)
        ax.set_xlabel("Mean marginal coverage")
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=13)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "figure_shift_aware_conformal_baselines.png", dpi=dpi)
    plt.close(fig)


def plot_shift_atcp_frontier(conformal: pd.DataFrame, paths: dict[str, Path], dpi: int) -> None:
    methods = [
        "adaptive_transport_conformal",
        "weighted_target_dominant_atcp",
        "shift_aware_source_trust_atcp",
        "safety_gated_shift_atcp",
        "risk_asymmetric_shift_atcp",
        "dp_shift_atcp",
        "multi_objective_safety_gated_atcp",
        "multi_objective_dp_shift_atcp",
        "target_only_conformal",
    ]
    labels = {
        "adaptive_transport_conformal": "ATCP",
        "weighted_target_dominant_atcp": "Weighted ATCP",
        "shift_aware_source_trust_atcp": "SHIFT-ATCP",
        "safety_gated_shift_atcp": "Safety-gated SHIFT",
        "risk_asymmetric_shift_atcp": "Risk-asymmetric SHIFT",
        "dp_shift_atcp": "DP-SHIFT",
        "multi_objective_safety_gated_atcp": "Multi-objective SHIFT",
        "multi_objective_dp_shift_atcp": "Multi-objective DP-SHIFT",
        "target_only_conformal": "Target-only",
    }
    data = conformal.loc[conformal["conformal_method"].isin(methods)].copy()
    if data.empty:
        return
    data["method_label"] = data["conformal_method"].map(labels)
    summary = data.groupby(["target_dataset", "confidence_level", "method_label"], as_index=False).agg(
        coverage=("marginal_coverage", "mean"),
        set_size=("average_prediction_set_size", "mean"),
        referral_rate=("referral_rate", "mean"),
        clinical_cost=("clinical_cost", "mean"),
    )
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
    for ax, target in zip(axes, ["NHANES", "Pima"]):
        sub = summary.loc[summary["target_dataset"].eq(target)]
        sns.lineplot(data=sub, x="set_size", y="coverage", hue="method_label", marker="o", linewidth=2.4, ax=ax)
        ax.axhline(PRIMARY_CONFIDENCE, color="black", linestyle="--", linewidth=2)
        ax.set_title(target)
        ax.set_xlabel("Average prediction set size")
        ax.set_ylabel("Marginal coverage")
        ax.tick_params(axis="both", labelsize=13)
        ax.legend(fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "figure_shift_atcp_coverage_efficiency_frontier.png", dpi=dpi)
    plt.close(fig)

    primary = summary.loc[summary["confidence_level"].eq(PRIMARY_CONFIDENCE)].copy()
    if primary.empty:
        return

    plot_order = [
        "ATCP",
        "Target-only",
        "Weighted ATCP",
        "SHIFT-ATCP",
        "Safety-gated SHIFT",
        "DP-SHIFT",
        "Risk-asymmetric SHIFT",
        "Multi-objective DP-SHIFT",
    ]
    compact_labels = {
        "ATCP": "Old ATCP",
        "Target-only": "Target-only",
        "Weighted ATCP": "Weighted ATCP",
        "SHIFT-ATCP": "SHIFT-ATCP",
        "Safety-gated SHIFT": "Safety-gated SHIFT",
        "DP-SHIFT": "DP-SHIFT",
        "Risk-asymmetric SHIFT": "Risk-asymmetric",
        "Multi-objective DP-SHIFT": "Multi-objective DP",
    }
    primary = primary.loc[primary["method_label"].isin(plot_order)].copy()
    primary["method_display"] = primary["method_label"].map(compact_labels)
    primary["method_display"] = pd.Categorical(
        primary["method_display"],
        categories=[compact_labels[m] for m in plot_order if m in set(primary["method_label"])],
        ordered=True,
    )

    palette = sns.color_palette("Set2", n_colors=len(plot_order))
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=True)
    metric_specs = [
        ("referral_rate", "Referral rate", 0),
        ("clinical_cost", "Clinical cost", 1),
    ]
    for col, target in enumerate(["NHANES", "Pima"]):
        target_data = primary.loc[primary["target_dataset"].eq(target)].sort_values("method_display")
        for metric, xlabel, row in metric_specs:
            ax = axes[row, col]
            sns.barplot(
                data=target_data,
                y="method_display",
                x=metric,
                order=target_data["method_display"].cat.categories,
                palette=palette,
                ax=ax,
                orient="h",
                errorbar=None,
            )
            xmax = max(float(target_data[metric].max()) * 1.18, 0.05)
            ax.set_xlim(0, xmax)
            ax.set_title(f"{target}: {xlabel}", fontsize=18, weight="bold")
            ax.set_xlabel(xlabel, fontsize=15)
            ax.set_ylabel("")
            ax.tick_params(axis="x", labelsize=13)
            ax.tick_params(axis="y", labelsize=13)
            for patch in ax.patches:
                width = patch.get_width()
                y = patch.get_y() + patch.get_height() / 2
                ax.text(width + xmax * 0.015, y, f"{width:.3f}", va="center", ha="left", fontsize=11)
    fig.suptitle("Clinical Referral and Cost Profile of SHIFT-ATCP Variants", fontsize=22, weight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(paths["figures"] / "figure_shift_atcp_clinical_referral_cost.png", dpi=dpi)
    plt.close(fig)


def plot_source_trust_stress(source_trust_stress: pd.DataFrame, paths: dict[str, Path], dpi: int) -> None:
    if source_trust_stress.empty:
        return
    summary = source_trust_stress.groupby(["target_dataset", "target_mix_fraction"], as_index=False).agg(
        source_weight=("shift_formula_source_weight", "mean"),
        coverage=("marginal_coverage", "mean"),
        diabetes_coverage=("class_1_conditional_coverage", "mean"),
        set_size=("average_prediction_set_size", "mean"),
    )
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
    for ax, target in zip(axes, ["NHANES", "Pima"]):
        sub = summary.loc[summary["target_dataset"].eq(target)]
        ax.plot(sub["target_mix_fraction"], sub["source_weight"], marker="o", linewidth=2.5, label="Source trust weight")
        ax.set_title(target)
        ax.set_xlabel("Target fraction in simulated calibration/test data")
        ax.set_ylabel("Source trust weight")
        ax.tick_params(axis="both", labelsize=13)
        ax2 = ax.twinx()
        ax2.plot(sub["target_mix_fraction"], sub["coverage"], marker="s", linewidth=2.5, color="#F58518", label="Marginal coverage")
        ax2.plot(sub["target_mix_fraction"], sub["diabetes_coverage"], marker="^", linewidth=2.5, color="#54A24B", label="Diabetes coverage")
        ax2.axhline(PRIMARY_CONFIDENCE, color="black", linestyle="--", linewidth=1.8)
        ax2.set_ylabel("Coverage")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, fontsize=10, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "figure_source_trust_stress_test.png", dpi=dpi)
    plt.close(fig)


def clinical_tradeoff_table(clinical_summary: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "adaptive_transport_conformal",
        "weighted_target_dominant_atcp",
        "shift_aware_source_trust_atcp",
        "safety_gated_shift_atcp",
        "dp_shift_atcp",
        "risk_asymmetric_shift_atcp",
        "multi_objective_dp_shift_atcp",
        "target_only_conformal",
    ]
    method_labels = {
        "adaptive_transport_conformal": "Old ATCP",
        "weighted_target_dominant_atcp": "Weighted ATCP",
        "shift_aware_source_trust_atcp": "SHIFT-ATCP",
        "safety_gated_shift_atcp": "Safety-gated SHIFT",
        "dp_shift_atcp": "DP-SHIFT",
        "risk_asymmetric_shift_atcp": "Risk-asymmetric SHIFT",
        "multi_objective_dp_shift_atcp": "Multi-objective DP-SHIFT",
        "target_only_conformal": "Target-only",
    }
    table = clinical_summary.loc[clinical_summary["conformal_method"].isin(methods)].copy()
    table["method_label"] = table["conformal_method"].map(method_labels)
    table["coverage_error_from_90"] = (table["mean_coverage"] - PRIMARY_CONFIDENCE).abs()
    keep = [
        "target_dataset",
        "method_label",
        "conformal_method",
        "mean_coverage",
        "mean_class_1_coverage",
        "mean_set_size",
        "mean_referral_rate",
        "mean_clinical_cost",
        "mean_class_gap",
        "coverage_error_from_90",
        "mean_source_weight",
        "fallback_rate",
    ]
    return table[[col for col in keep if col in table.columns]].sort_values(
        ["target_dataset", "mean_class_1_coverage", "mean_clinical_cost"],
        ascending=[True, False, True],
    )


def plot_clinical_tradeoff(tradeoff: pd.DataFrame, paths: dict[str, Path], dpi: int) -> None:
    if tradeoff.empty:
        return
    sns.set_theme(style="whitegrid", context="talk")
    method_order = [
        "Old ATCP",
        "Weighted ATCP",
        "SHIFT-ATCP",
        "Safety-gated SHIFT",
        "Target-only",
        "Multi-objective DP-SHIFT",
        "DP-SHIFT",
        "Risk-asymmetric SHIFT",
    ]
    data = tradeoff.copy()
    data["method_label"] = pd.Categorical(data["method_label"], categories=method_order, ordered=True)
    data = data.sort_values(["method_label", "target_dataset"])
    target_palette = {"NHANES": "#4C78A8", "Pima": "#F58518"}
    target_offsets = {"NHANES": -0.14, "Pima": 0.14}
    metrics = [
        ("mean_coverage", "Marginal coverage", "Higher is better", (0.74, 0.925), PRIMARY_CONFIDENCE),
        ("mean_class_1_coverage", "Diabetes-positive coverage", "Higher is safer", (0.45, 0.98), PRIMARY_CONFIDENCE),
        ("mean_set_size", "Average set size", "Lower is more efficient", None, None),
        ("mean_clinical_cost", "Clinical cost", "Lower is better", None, None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=True)
    y_positions = np.arange(len(method_order))
    for ax, (metric, title, subtitle, xlim, reference) in zip(axes.ravel(), metrics):
        for target in ["NHANES", "Pima"]:
            sub = data.loc[data["target_dataset"].eq(target)]
            y = sub["method_label"].cat.codes.to_numpy(dtype=float) + target_offsets[target]
            ax.scatter(
                sub[metric],
                y,
                s=170,
                color=target_palette[target],
                edgecolor="black",
                linewidth=0.8,
                label=target,
                alpha=0.92,
                zorder=3,
            )
        if reference is not None:
            ax.axvline(reference, color="black", linestyle="--", linewidth=2, alpha=0.85)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_title(f"{title}\n{subtitle}", fontsize=16)
        ax.set_xlabel(title)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(method_order)
        ax.invert_yaxis()
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="x", alpha=0.35)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Clinical Reliability Trade-off of SHIFT-ATCP Variants", fontsize=22, y=1.035)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(paths["figures"] / "figure_clinical_reliability_tradeoff_shift_atcp_variants.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    paths = ensure_tree(Path(args.outdir))
    base.set_plot_style()

    source = pd.read_csv(input_dir / "data" / "kaggle_harmonized.csv")
    nhanes = pd.read_csv(input_dir / "data" / "nhanes_harmonized.csv")
    pima = pd.read_csv(input_dir / "data" / "pima_harmonized.csv")
    datasets = {"Kaggle": source, "NHANES": nhanes, "Pima": pima}
    for name, df in datasets.items():
        df.to_csv(paths["data"] / f"{name.lower()}_harmonized.csv", index=False)

    train, source_calib, source_test = source_split(source, args.seed)
    source_calib_fit, source_calib_select = calibration_fit_select_split(source_calib, args.seed)
    nhanes_calib, nhanes_test = external_split(nhanes, args.seed)
    pima_calib, pima_test = external_split(pima, args.seed)
    external = {"NHANES": (nhanes_calib, nhanes_test), "Pima": (pima_calib, pima_test)}

    split_rows = [
        {"dataset": "Kaggle", **split_counts(train, "source_train"), "random_seed": args.seed, "stratified_by": TARGET},
        {"dataset": "Kaggle", **split_counts(source_calib, "source_calibration"), "random_seed": args.seed, "stratified_by": TARGET},
        {"dataset": "Kaggle", **split_counts(source_test, "source_internal_test"), "random_seed": args.seed, "stratified_by": TARGET},
        {"dataset": "NHANES", **split_counts(nhanes_calib, "target_calibration"), "random_seed": args.seed, "stratified_by": TARGET},
        {"dataset": "NHANES", **split_counts(nhanes_test, "target_test"), "random_seed": args.seed, "stratified_by": TARGET},
        {"dataset": "Pima", **split_counts(pima_calib, "target_calibration"), "random_seed": args.seed, "stratified_by": TARGET},
        {"dataset": "Pima", **split_counts(pima_test, "target_test"), "random_seed": args.seed, "stratified_by": TARGET},
    ]
    pd.DataFrame(split_rows).to_csv(paths["tables"] / "table_01_split_construction.csv", index=False)
    pd.read_csv(input_dir / "tables" / "table_02_dataset_summary.csv").to_csv(paths["tables"] / "table_02_dataset_summary.csv", index=False)
    psi_table = pd.read_csv(input_dir / "tables" / "table_03_dataset_shift_metrics.csv")
    psi_table.to_csv(paths["tables"] / "table_03_dataset_shift_metrics.csv", index=False)

    tuning_rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    source_calibration_selection_rows: list[dict[str, Any]] = []
    target_calibration_selection_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    conformal_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    shift_source_trust_rows: list[dict[str, Any]] = []
    source_trust_stress_rows: list[dict[str, Any]] = []
    model_cache: dict[tuple[str, str], Pipeline] = {}
    source_calib_prob_cache: dict[tuple[str, str], tuple[str, base.ProbabilityCalibrator, np.ndarray]] = {}

    source_prev = float(train[TARGET].mean())
    model_names = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]

    for feature_set, features in FEATURE_SETS.items():
        shift_context: dict[str, dict[str, Any]] = {}
        for target_name, (target_calib, _) in external.items():
            mean_psi = mean_psi_for_features(psi_table, target_name, features)
            domain_auc = domain_shift_auc(source_calib, target_calib, features, args.seed)
            target_prev = float(target_calib[TARGET].mean())
            prevalence_gap = float(abs(target_prev - source_prev))
            shift_weight = shift_aware_source_weight(
                mean_psi=mean_psi,
                domain_auc=domain_auc,
                target_n=len(target_calib),
                prevalence_gap=prevalence_gap,
            )
            row = {
                "target_dataset": target_name,
                "feature_set": feature_set,
                "source_calibration_n": int(len(source_calib)),
                "target_calibration_n": int(len(target_calib)),
                "source_diabetes_prevalence": source_prev,
                "target_diabetes_prevalence": target_prev,
                "prevalence_gap": prevalence_gap,
                "mean_psi": mean_psi,
                "domain_classifier_auc": domain_auc,
                "shift_aware_source_weight": shift_weight,
                "source_trust_formula": "exp(-2*mean_psi)*exp(-3*max(domain_auc-0.5,0))*target_n^-0.5*exp(-2*prevalence_gap)",
            }
            shift_context[target_name] = row
            shift_source_trust_rows.append(row)

        X_train = base.prepare_X(train, features)
        y_train = train[TARGET]
        X_source_test = base.prepare_X(source_test, features)
        y_source_test = source_test[TARGET]
        X_source_calib_fit = base.prepare_X(source_calib_fit, features)
        X_source_calib_select = base.prepare_X(source_calib_select, features)
        X_source_calib_full = base.prepare_X(source_calib, features)
        y_source_calib_fit = source_calib_fit[TARGET]
        y_source_calib_select = source_calib_select[TARGET]
        y_source_calib_full = source_calib[TARGET]

        for model_name in model_names:
            print(f"Fitting tuned source model: {feature_set} / {model_name}", flush=True)
            model, rows = tune_and_fit_model(model_name, X_train, y_train, args.seed, args.quick)
            for row in rows:
                row["feature_set"] = feature_set
            tuning_rows.extend(rows)
            model_cache[(feature_set, model_name)] = model
            joblib.dump(model, paths["models"] / f"{feature_set}_{model_name.lower().replace(' ', '_')}_tuned.joblib")

            source_test_raw = base.positive_probabilities(model, X_source_test)
            internal_rows.append(base.classification_metrics(y_source_test, source_test_raw, {
                "dataset": "Kaggle_internal_test",
                "feature_set": feature_set,
                "model": model_name,
                "calibration_method": "uncalibrated",
                "hyperparameter_protocol": "equal_budget_source_validation",
            }))

            raw_fit = base.positive_probabilities(model, X_source_calib_fit)
            raw_select = base.positive_probabilities(model, X_source_calib_select)
            raw_full = base.positive_probabilities(model, X_source_calib_full)
            best_source_method, _, source_rows = select_probability_calibrator(
                raw_fit, y_source_calib_fit, raw_select, y_source_calib_select,
                {"feature_set": feature_set, "model": model_name, "selection_split": "source_calibration_select"},
            )
            source_calibration_selection_rows.extend(source_rows)
            final_source_cal = base.ProbabilityCalibrator(best_source_method).fit(raw_full, y_source_calib_full)
            source_calib_probs = final_source_cal.transform(raw_full)
            source_calib_prob_cache[(feature_set, model_name)] = (best_source_method, final_source_cal, source_calib_probs)

            for target_name, (target_calib, target_test) in external.items():
                print(f"  Evaluating target: {target_name}", flush=True)
                X_target_calib = base.prepare_X(target_calib, features)
                X_target_test = base.prepare_X(target_test, features)
                y_target_calib = target_calib[TARGET]
                y_target_test = target_test[TARGET]
                target_calib_raw = base.positive_probabilities(model, X_target_calib)
                target_test_raw = base.positive_probabilities(model, X_target_test)
                target_prev = float(y_target_calib.mean())

                target_fit, target_select = calibration_fit_select_split(target_calib, args.seed)
                X_target_fit = base.prepare_X(target_fit, features)
                X_target_select = base.prepare_X(target_select, features)
                target_fit_raw = base.positive_probabilities(model, X_target_fit)
                target_select_raw = base.positive_probabilities(model, X_target_select)
                adaptive_method, _, target_rows = select_probability_calibrator(
                    target_fit_raw, target_fit[TARGET], target_select_raw, target_select[TARGET],
                    {"target_dataset": target_name, "feature_set": feature_set, "model": model_name, "selection_split": "target_calibration_select"},
                )
                target_calibration_selection_rows.extend(target_rows)
                adaptive_cal = base.ProbabilityCalibrator(adaptive_method).fit(target_calib_raw, y_target_calib)
                adaptive_gate_cal = base.ProbabilityCalibrator(adaptive_method).fit(target_fit_raw, target_fit[TARGET])

                fixed_cal = base.ProbabilityCalibrator(best_source_method).fit(target_calib_raw, y_target_calib)
                fixed_gate_cal = base.ProbabilityCalibrator(best_source_method).fit(target_fit_raw, target_fit[TARGET])
                label_shift_calib = label_shift_adjust(target_calib_raw, source_prev, target_prev)
                label_shift_test = label_shift_adjust(target_test_raw, source_prev, target_prev)
                label_shift_source = label_shift_adjust(raw_full, source_prev, target_prev)
                label_shift_fit = label_shift_adjust(target_fit_raw, source_prev, target_prev)
                label_shift_select = label_shift_adjust(target_select_raw, source_prev, target_prev)

                scenario_defs = {
                    "naive_transfer": {
                        "method": "uncalibrated",
                        "p_test": target_test_raw,
                        "p_calib": target_calib_raw,
                        "p_source": raw_full,
                        "p_fit": target_fit_raw,
                        "p_select": target_select_raw,
                    },
                    "label_shift_adjusted": {
                        "method": "label_shift",
                        "p_test": label_shift_test,
                        "p_calib": label_shift_calib,
                        "p_source": label_shift_source,
                        "p_fit": label_shift_fit,
                        "p_select": label_shift_select,
                    },
                    "fixed_recalibration_source_selected": {
                        "method": best_source_method,
                        "p_test": fixed_cal.transform(target_test_raw),
                        "p_calib": fixed_cal.transform(target_calib_raw),
                        "p_source": fixed_cal.transform(raw_full),
                        "p_fit": fixed_gate_cal.transform(target_fit_raw),
                        "p_select": fixed_gate_cal.transform(target_select_raw),
                    },
                    "adaptive_calibration_target_selected": {
                        "method": adaptive_method,
                        "p_test": adaptive_cal.transform(target_test_raw),
                        "p_calib": adaptive_cal.transform(target_calib_raw),
                        "p_source": adaptive_cal.transform(raw_full),
                        "p_fit": adaptive_gate_cal.transform(target_fit_raw),
                        "p_select": adaptive_gate_cal.transform(target_select_raw),
                    },
                }
                source_weights = estimate_density_ratio_weights(source_calib, target_calib, features, args.seed)
                shift_info = shift_context[target_name]
                shift_label = {
                    "mean_psi": shift_info["mean_psi"],
                    "domain_classifier_auc": shift_info["domain_classifier_auc"],
                    "prevalence_gap": shift_info["prevalence_gap"],
                    "shift_formula_source_weight": shift_info["shift_aware_source_weight"],
                }
                for scenario, scenario_values in scenario_defs.items():
                    method = scenario_values["method"]
                    p_test = scenario_values["p_test"]
                    p_calib = scenario_values["p_calib"]
                    label = {
                        "target_dataset": target_name,
                        "feature_set": feature_set,
                        "model": model_name,
                        "scenario": scenario,
                        "calibration_method": method,
                    }
                    scenario_rows.append(base.classification_metrics(y_target_test, p_test, label))
                    conformal_rows.extend(conformal_rows_for_scenario(
                        y_target_test,
                        p_test,
                        y_source_calib_full,
                        scenario_values["p_source"],
                        y_target_calib,
                        p_calib,
                        target_fit[TARGET],
                        scenario_values["p_fit"],
                        target_select[TARGET],
                        scenario_values["p_select"],
                        source_weights,
                        shift_info["shift_aware_source_weight"],
                        shift_label,
                        label,
                        args.bootstrap_repeats,
                        args.seed,
                    ))

                p_source_calib_adaptive = adaptive_cal.transform(raw_full)
                p_target_calib_adaptive = adaptive_cal.transform(target_calib_raw)
                p_source_test_adaptive = adaptive_cal.transform(source_test_raw)
                p_target_test_adaptive = adaptive_cal.transform(target_test_raw)
                for target_mix in SOURCE_TRUST_MIX_GRID:
                    rng = np.random.default_rng(args.seed + len(source_trust_stress_rows) + int(target_mix * 1000))
                    n_calib = len(target_calib)
                    n_test = len(target_test)
                    n_target_calib = int(round(n_calib * target_mix))
                    n_source_calib = n_calib - n_target_calib
                    n_target_test = int(round(n_test * target_mix))
                    n_source_test = n_test - n_target_test
                    source_calib_idx = sampled_indices(len(source_calib), n_source_calib, rng)
                    target_calib_idx = sampled_indices(len(target_calib), n_target_calib, rng)
                    source_test_idx = sampled_indices(len(source_test), n_source_test, rng)
                    target_test_idx = sampled_indices(len(target_test), n_target_test, rng)

                    mixed_calib_df = pd.concat([
                        source_calib.iloc[source_calib_idx],
                        target_calib.iloc[target_calib_idx],
                    ], ignore_index=True)
                    mixed_y_calib = pd.concat([
                        y_source_calib_full.iloc[source_calib_idx].reset_index(drop=True),
                        y_target_calib.iloc[target_calib_idx].reset_index(drop=True),
                    ], ignore_index=True)
                    mixed_p_calib = np.concatenate([
                        positive_from_probs(p_source_calib_adaptive)[source_calib_idx],
                        positive_from_probs(p_target_calib_adaptive)[target_calib_idx],
                    ])
                    mixed_y_test = pd.concat([
                        y_source_test.iloc[source_test_idx].reset_index(drop=True),
                        y_target_test.iloc[target_test_idx].reset_index(drop=True),
                    ], ignore_index=True)
                    mixed_p_test = np.concatenate([
                        positive_from_probs(p_source_test_adaptive)[source_test_idx],
                        positive_from_probs(p_target_test_adaptive)[target_test_idx],
                    ])

                    stress_mean_psi = compute_mean_psi(source_calib, mixed_calib_df, features)
                    stress_domain_auc = domain_shift_auc(source_calib, mixed_calib_df, features, args.seed)
                    stress_prevalence_gap = float(abs(float(mixed_y_calib.mean()) - source_prev))
                    stress_source_weight = shift_aware_source_weight(
                        mean_psi=stress_mean_psi,
                        domain_auc=stress_domain_auc,
                        target_n=len(mixed_y_calib),
                        prevalence_gap=stress_prevalence_gap,
                    )
                    q_stress = weighted_atcp_quantile(
                        p_source_calib_adaptive,
                        y_source_calib_full,
                        mixed_p_calib,
                        mixed_y_calib,
                        PRIMARY_CONFIDENCE,
                        source_weight=stress_source_weight,
                    )
                    sets_stress = base.conformal_sets(as_probability_matrix(mixed_p_test), q_stress)
                    stress_metrics = extended_conformal_metrics(
                        mixed_y_test,
                        sets_stress,
                        {"confidence_level": PRIMARY_CONFIDENCE},
                        probs=as_probability_matrix(mixed_p_test),
                        bootstrap_repeats=0,
                        seed=args.seed,
                    )
                    source_trust_stress_rows.append({
                        "target_dataset": target_name,
                        "feature_set": feature_set,
                        "model": model_name,
                        "calibration_method": adaptive_method,
                        "target_mix_fraction": target_mix,
                        "source_mix_fraction": 1.0 - target_mix,
                        "mixed_calibration_n": int(len(mixed_y_calib)),
                        "mixed_test_n": int(len(mixed_y_test)),
                        "mixed_diabetes_prevalence": float(mixed_y_calib.mean()),
                        "mean_psi": stress_mean_psi,
                        "domain_classifier_auc": stress_domain_auc,
                        "prevalence_gap": stress_prevalence_gap,
                        "shift_formula_source_weight": stress_source_weight,
                        "marginal_coverage": stress_metrics["marginal_coverage"],
                        "class_0_conditional_coverage": stress_metrics["class_0_conditional_coverage"],
                        "class_1_conditional_coverage": stress_metrics["class_1_conditional_coverage"],
                        "class_coverage_gap_abs": stress_metrics["class_coverage_gap_abs"],
                        "average_prediction_set_size": stress_metrics["average_prediction_set_size"],
                        "referral_rate": stress_metrics["referral_rate"],
                        "clinical_cost": stress_metrics["clinical_cost"],
                    })

                sizes = [50, 100, 200, 500, 750, len(target_calib)] if target_name == "NHANES" else [50, 100, 200, 300, len(target_calib)]
                for size in sizes:
                    if size > len(target_calib):
                        continue
                    for repeat in range(args.ablation_repeats):
                        sample_seed = args.seed + 1000 * repeat + size
                        sub_calib = stratified_subsample(target_calib, size, sample_seed)
                        sub_fit, sub_select = calibration_fit_select_split(sub_calib, sample_seed)
                        X_sub_fit = base.prepare_X(sub_fit, features)
                        X_sub_select = base.prepare_X(sub_select, features)
                        sub_fit_raw = base.positive_probabilities(model, X_sub_fit)
                        sub_select_raw = base.positive_probabilities(model, X_sub_select)
                        sub_method, _, _ = select_probability_calibrator(
                            sub_fit_raw, sub_fit[TARGET], sub_select_raw, sub_select[TARGET],
                            {"target_dataset": target_name, "feature_set": feature_set, "model": model_name, "selection_split": "ablation"},
                        )
                        X_sub_full = base.prepare_X(sub_calib, features)
                        sub_full_raw = base.positive_probabilities(model, X_sub_full)
                        sub_cal = base.ProbabilityCalibrator(sub_method).fit(sub_full_raw, sub_calib[TARGET])
                        sub_gate_cal = base.ProbabilityCalibrator(sub_method).fit(sub_fit_raw, sub_fit[TARGET])
                        p_test_sub = sub_cal.transform(target_test_raw)
                        p_calib_sub = sub_cal.transform(sub_full_raw)
                        p_source_sub = sub_cal.transform(raw_full)
                        p_source_gate = sub_gate_cal.transform(raw_full)
                        p_fit_gate = sub_gate_cal.transform(sub_fit_raw)
                        p_select_gate = sub_gate_cal.transform(sub_select_raw)
                        selected_weight, gate_coverage, gate_set_size = select_atcp_source_weight(
                            y_source_calib_full,
                            p_source_gate,
                            sub_fit[TARGET],
                            p_fit_gate,
                            sub_select[TARGET],
                            p_select_gate,
                            PRIMARY_CONFIDENCE,
                        )
                        sub_prevalence_gap = float(abs(float(sub_calib[TARGET].mean()) - source_prev))
                        shift_weight_sub = shift_aware_source_weight(
                            mean_psi=shift_info["mean_psi"],
                            domain_auc=shift_info["domain_classifier_auc"],
                            target_n=len(sub_calib),
                            prevalence_gap=sub_prevalence_gap,
                        )
                        q_target = base.conformal_quantile(as_probability_matrix(p_calib_sub), sub_calib[TARGET], PRIMARY_CONFIDENCE)
                        q_atcp = weighted_atcp_quantile(
                            p_source_sub,
                            y_source_calib_full,
                            p_calib_sub,
                            sub_calib[TARGET],
                            PRIMARY_CONFIDENCE,
                            source_weight=1.0,
                        )
                        q_weighted_atcp = weighted_atcp_quantile(
                            p_source_sub,
                            y_source_calib_full,
                            p_calib_sub,
                            sub_calib[TARGET],
                            PRIMARY_CONFIDENCE,
                            source_weight=selected_weight,
                        )
                        q_shift_atcp = weighted_atcp_quantile(
                            p_source_sub,
                            y_source_calib_full,
                            p_calib_sub,
                            sub_calib[TARGET],
                            PRIMARY_CONFIDENCE,
                            source_weight=shift_weight_sub,
                        )
                        q_by_class = weighted_class_conditional_quantiles(
                            p_source_sub,
                            y_source_calib_full,
                            p_calib_sub,
                            sub_calib[TARGET],
                            PRIMARY_CONFIDENCE,
                            source_weight=selected_weight,
                        )
                        q_risk_shift = weighted_risk_asymmetric_quantiles(
                            p_source_sub,
                            y_source_calib_full,
                            p_calib_sub,
                            sub_calib[TARGET],
                            PRIMARY_CONFIDENCE,
                            source_weight=shift_weight_sub,
                        )
                        shift_select_q = weighted_atcp_quantile(
                            p_source_gate,
                            y_source_calib_full,
                            p_fit_gate,
                            sub_fit[TARGET],
                            PRIMARY_CONFIDENCE,
                            source_weight=shift_weight_sub,
                        )
                        shift_select_sets = base.conformal_sets(as_probability_matrix(p_select_gate), shift_select_q)
                        shift_gate_metrics = extended_conformal_metrics(
                            sub_select[TARGET],
                            shift_select_sets,
                            {"confidence_level": PRIMARY_CONFIDENCE},
                            bootstrap_repeats=0,
                            seed=sample_seed,
                        )
                        multi = select_multi_objective_weight(
                            y_source_calib_full,
                            p_source_gate,
                            sub_fit[TARGET],
                            p_fit_gate,
                            sub_select[TARGET],
                            p_select_gate,
                            PRIMARY_CONFIDENCE,
                            shift_weight_sub,
                            sample_seed,
                        )
                        dp_selection = select_diabetes_protected_params(
                            y_source_calib_full,
                            p_source_gate,
                            sub_fit[TARGET],
                            p_fit_gate,
                            sub_select[TARGET],
                            p_select_gate,
                            PRIMARY_CONFIDENCE,
                            shift_weight_sub,
                            sample_seed,
                        )
                        dp_class_confidences = {
                            0: dp_selection["class_0_target_confidence"],
                            1: dp_selection["class_1_target_confidence"],
                        }
                        q_dp = weighted_class_specific_quantiles(
                            p_source_sub,
                            y_source_calib_full,
                            p_calib_sub,
                            sub_calib[TARGET],
                            dp_class_confidences,
                            source_weight=shift_weight_sub,
                        )
                        test_prob_matrix = as_probability_matrix(p_test_sub)
                        method_sets = [
                            ("adaptive_transport_conformal", base.conformal_sets(test_prob_matrix, q_atcp), 1.0, False, "unweighted_atcp", np.nan, np.nan, "unweighted", np.nan, np.nan),
                            ("weighted_target_dominant_atcp", base.conformal_sets(test_prob_matrix, q_weighted_atcp), selected_weight, False, "weighted_atcp", gate_coverage, gate_set_size, "grid_selected", np.nan, np.nan),
                            ("shift_aware_source_trust_atcp", base.conformal_sets(test_prob_matrix, q_shift_atcp), shift_weight_sub, False, "shift_atcp", np.nan, np.nan, "shift_formula", np.nan, np.nan),
                            ("class_conditional_weighted_atcp", class_conditional_sets(test_prob_matrix, q_by_class), selected_weight, False, "class_conditional_weighted_atcp", gate_coverage, gate_set_size, "grid_selected", np.nan, np.nan),
                            ("risk_asymmetric_shift_atcp", class_conditional_sets(test_prob_matrix, q_risk_shift), shift_weight_sub, False, "risk_asymmetric_shift_atcp", np.nan, np.nan, "shift_formula", risk_asymmetric_confidences(PRIMARY_CONFIDENCE)[0], risk_asymmetric_confidences(PRIMARY_CONFIDENCE)[1]),
                            ("dp_shift_atcp", class_conditional_sets(test_prob_matrix, q_dp), shift_weight_sub, False, "dp_shift_atcp", dp_selection["validation_coverage"], dp_selection["validation_set_size"], "shift_formula", dp_class_confidences[0], dp_class_confidences[1]),
                        ]
                        if gate_coverage < PRIMARY_CONFIDENCE:
                            method_sets.append(("safety_gated_atcp", base.conformal_sets(test_prob_matrix, q_target), selected_weight, True, "target_only_fallback", gate_coverage, gate_set_size, "grid_selected", np.nan, np.nan))
                        else:
                            method_sets.append(("safety_gated_atcp", base.conformal_sets(test_prob_matrix, q_weighted_atcp), selected_weight, False, "weighted_atcp_accepted", gate_coverage, gate_set_size, "grid_selected", np.nan, np.nan))
                        if shift_gate_metrics["marginal_coverage"] < PRIMARY_CONFIDENCE:
                            method_sets.append(("safety_gated_shift_atcp", base.conformal_sets(test_prob_matrix, q_target), shift_weight_sub, True, "target_only_fallback", shift_gate_metrics["marginal_coverage"], shift_gate_metrics["average_prediction_set_size"], "shift_formula", np.nan, np.nan))
                        else:
                            method_sets.append(("safety_gated_shift_atcp", base.conformal_sets(test_prob_matrix, q_shift_atcp), shift_weight_sub, False, "shift_atcp_accepted", shift_gate_metrics["marginal_coverage"], shift_gate_metrics["average_prediction_set_size"], "shift_formula", np.nan, np.nan))
                        if multi["candidate_type"] == "target_only" or multi["validation_coverage"] < PRIMARY_CONFIDENCE:
                            method_sets.append(("multi_objective_safety_gated_atcp", base.conformal_sets(test_prob_matrix, q_target), multi["source_weight"], True, "target_only_fallback", multi["validation_coverage"], multi["validation_set_size"], "multi_objective", np.nan, np.nan))
                        else:
                            q_multi = weighted_atcp_quantile(
                                p_source_sub,
                                y_source_calib_full,
                                p_calib_sub,
                                sub_calib[TARGET],
                                PRIMARY_CONFIDENCE,
                                source_weight=multi["source_weight"],
                            )
                            method_sets.append(("multi_objective_safety_gated_atcp", base.conformal_sets(test_prob_matrix, q_multi), multi["source_weight"], False, "multi_objective_weighted_atcp", multi["validation_coverage"], multi["validation_set_size"], "multi_objective", np.nan, np.nan))
                        if dp_selection["validation_coverage"] < PRIMARY_CONFIDENCE:
                            method_sets.append(("multi_objective_dp_shift_atcp", base.conformal_sets(test_prob_matrix, q_target), shift_weight_sub, True, "target_only_fallback", dp_selection["validation_coverage"], dp_selection["validation_set_size"], "shift_formula", dp_class_confidences[0], dp_class_confidences[1]))
                        else:
                            method_sets.append(("multi_objective_dp_shift_atcp", class_conditional_sets(test_prob_matrix, q_dp), shift_weight_sub, False, "diabetes_protected_shift_atcp", dp_selection["validation_coverage"], dp_selection["validation_set_size"], "shift_formula", dp_class_confidences[0], dp_class_confidences[1]))
                        metrics = base.classification_metrics(y_target_test, p_test_sub, {
                            "target_dataset": target_name,
                            "feature_set": feature_set,
                            "model": model_name,
                            "repeat": repeat,
                            "calibration_size": int(size),
                            "calibration_method": sub_method,
                            "confidence_level": PRIMARY_CONFIDENCE,
                        })
                        for conformal_method, sets, method_weight, fallback_used, safety_decision, method_gate_coverage, method_gate_set_size, source_weight_strategy, class_0_target_confidence, class_1_target_confidence in method_sets:
                            conf = extended_conformal_metrics(y_target_test, sets, {
                                "confidence_level": PRIMARY_CONFIDENCE,
                            }, probs=test_prob_matrix, bootstrap_repeats=0, seed=sample_seed)
                            ablation_rows.append({
                                **metrics,
                                "conformal_method": conformal_method,
                                "class_0_target_confidence": class_0_target_confidence,
                                "class_1_target_confidence": class_1_target_confidence,
                                "selected_source_weight": method_weight,
                                "source_weight_strategy": source_weight_strategy,
                                "gate_validation_coverage": method_gate_coverage,
                                "gate_validation_set_size": method_gate_set_size,
                                "fallback_used": fallback_used,
                                "safety_decision": safety_decision,
                                "mean_psi": shift_info["mean_psi"],
                                "domain_classifier_auc": shift_info["domain_classifier_auc"],
                                "prevalence_gap": sub_prevalence_gap,
                                "shift_formula_source_weight": shift_weight_sub,
                                "marginal_coverage": conf["marginal_coverage"],
                                "class_0_conditional_coverage": conf["class_0_conditional_coverage"],
                                "class_1_conditional_coverage": conf["class_1_conditional_coverage"],
                                "class_coverage_gap_abs": conf["class_coverage_gap_abs"],
                                "clinical_cost": conf["clinical_cost"],
                                "single_label_rate": conf["single_label_rate"],
                                "referral_rate": conf["referral_rate"],
                                "uncertain_set_rate": conf["uncertain_set_rate"],
                                "diabetes_prevalence_in_referred": conf["diabetes_prevalence_in_referred"],
                                "mean_confidence_referred": conf.get("mean_confidence_referred", np.nan),
                                "mean_confidence_non_referred": conf.get("mean_confidence_non_referred", np.nan),
                                "non_referred_error_rate": conf.get("non_referred_error_rate", np.nan),
                                "average_prediction_set_size": conf["average_prediction_set_size"],
                                "coverage_gap": conf["coverage_gap"],
                            })

    tuning = pd.DataFrame(tuning_rows)
    internal = pd.DataFrame(internal_rows)
    source_calibration = pd.DataFrame(source_calibration_selection_rows)
    target_calibration = pd.DataFrame(target_calibration_selection_rows)
    scenarios = pd.DataFrame(scenario_rows)
    conformal = pd.DataFrame(conformal_rows)
    ablation = pd.DataFrame(ablation_rows)
    shift_source_trust = pd.DataFrame(shift_source_trust_rows).drop_duplicates()
    source_trust_stress = pd.DataFrame(source_trust_stress_rows)

    tuning.to_csv(paths["tables"] / "table_04_equal_budget_hyperparameter_tuning.csv", index=False)
    internal.to_csv(paths["tables"] / "table_05_tuned_internal_source_model_metrics.csv", index=False)
    source_calibration.to_csv(paths["tables"] / "table_06_leakage_safe_source_calibration_selection.csv", index=False)
    target_calibration.to_csv(paths["tables"] / "table_07_target_calibration_only_adaptive_selection.csv", index=False)
    scenarios.to_csv(paths["tables"] / "table_08_leakage_safe_external_transportability_scenarios.csv", index=False)
    conformal.to_csv(paths["tables"] / "table_09_shift_aware_conformal_comparison.csv", index=False)
    ablation.to_csv(paths["tables"] / "table_10_target_calibration_size_ablation.csv", index=False)

    primary_conformal = conformal.loc[conformal["confidence_level"].eq(PRIMARY_CONFIDENCE)].copy()
    scenario_summary = scenarios.merge(
        primary_conformal[[
            "target_dataset", "feature_set", "model", "scenario", "calibration_method", "conformal_method",
            "marginal_coverage", "average_prediction_set_size",
        ]],
        on=["target_dataset", "feature_set", "model", "scenario", "calibration_method"],
        how="left",
    )
    scenario_summary["coverage_gap_abs"] = (scenario_summary["marginal_coverage"] - PRIMARY_CONFIDENCE).abs()
    scenario_summary.sort_values(["target_dataset", "coverage_gap_abs", "ece"]).to_csv(
        paths["tables"] / "table_11_best_shift_aware_results_summary.csv",
        index=False,
    )
    shift_source_trust.to_csv(paths["tables"] / "table_12_shift_aware_source_trust_metrics.csv", index=False)

    frontier_cols = [
        "target_dataset", "feature_set", "model", "scenario", "calibration_method", "conformal_method",
        "confidence_level", "marginal_coverage", "coverage_gap", "average_prediction_set_size",
        "referral_rate", "clinical_cost", "class_0_conditional_coverage", "class_1_conditional_coverage",
        "class_coverage_gap_abs", "selected_source_weight", "source_weight_strategy",
        "diabetes_prevalence_in_referred", "mean_confidence_referred", "mean_confidence_non_referred",
        "non_referred_error_rate",
    ]
    conformal[[col for col in frontier_cols if col in conformal.columns]].to_csv(
        paths["tables"] / "table_13_uncertainty_efficiency_frontier.csv",
        index=False,
    )

    primary = conformal.loc[conformal["confidence_level"].eq(PRIMARY_CONFIDENCE)].copy()
    clinical_summary = primary.groupby(["target_dataset", "conformal_method"], as_index=False).agg(
        mean_coverage=("marginal_coverage", "mean"),
        mean_set_size=("average_prediction_set_size", "mean"),
        mean_referral_rate=("referral_rate", "mean"),
        mean_clinical_cost=("clinical_cost", "mean"),
        mean_class_0_coverage=("class_0_conditional_coverage", "mean"),
        mean_class_1_coverage=("class_1_conditional_coverage", "mean"),
        mean_class_gap=("class_coverage_gap_abs", "mean"),
        mean_source_weight=("selected_source_weight", "mean"),
        fallback_rate=("fallback_used", "mean"),
        mean_diabetes_prevalence_in_referred=("diabetes_prevalence_in_referred", "mean"),
        mean_confidence_referred=("mean_confidence_referred", "mean"),
        mean_confidence_non_referred=("mean_confidence_non_referred", "mean"),
        mean_non_referred_error_rate=("non_referred_error_rate", "mean"),
    )
    clinical_summary.to_csv(paths["tables"] / "table_14_clinical_reliability_summary.csv", index=False)
    tradeoff = clinical_tradeoff_table(clinical_summary)
    tradeoff.to_csv(paths["tables"] / "table_20_clinical_reliability_tradeoff_shift_atcp_variants.csv", index=False)

    ci_cols = [
        "target_dataset", "feature_set", "model", "scenario", "calibration_method", "conformal_method",
        "marginal_coverage", "coverage_bootstrap_mean", "coverage_ci_low", "coverage_ci_high",
        "average_prediction_set_size", "referral_rate", "clinical_cost",
    ]
    ci = primary.loc[primary.get("coverage_ci_low", pd.Series(index=primary.index, dtype=float)).notna()]
    ci[[col for col in ci_cols if col in ci.columns]].to_csv(
        paths["tables"] / "table_15_bootstrap_coverage_confidence_intervals.csv",
        index=False,
    )
    dp_selection_cols = [
        "target_dataset", "feature_set", "model", "scenario", "calibration_method", "conformal_method",
        "confidence_level", "class_0_target_confidence", "class_1_target_confidence",
        "gate_validation_coverage", "gate_validation_class_0_coverage", "gate_validation_class_1_coverage",
        "gate_validation_class_gap", "gate_validation_set_size", "gate_validation_clinical_cost",
        "gate_validation_referral_rate", "multi_objective_score", "selected_source_weight",
        "marginal_coverage", "class_0_conditional_coverage", "class_1_conditional_coverage",
        "class_coverage_gap_abs", "average_prediction_set_size", "referral_rate", "clinical_cost",
    ]
    conformal.loc[
        conformal["conformal_method"].isin(["dp_shift_atcp", "multi_objective_dp_shift_atcp"]),
        [col for col in dp_selection_cols if col in conformal.columns],
    ].to_csv(paths["tables"] / "table_16_diabetes_protected_shift_atcp_selection.csv", index=False)

    source_trust_stress.to_csv(paths["tables"] / "table_17_source_trust_stress_test.csv", index=False)

    stress_summary = source_trust_stress.groupby(
        ["target_dataset", "feature_set", "target_mix_fraction"],
        as_index=False,
    ).agg(
        mean_psi=("mean_psi", "mean"),
        mean_domain_auc=("domain_classifier_auc", "mean"),
        mean_source_weight=("shift_formula_source_weight", "mean"),
        mean_coverage=("marginal_coverage", "mean"),
        mean_diabetes_coverage=("class_1_conditional_coverage", "mean"),
        mean_set_size=("average_prediction_set_size", "mean"),
        mean_referral_rate=("referral_rate", "mean"),
        mean_clinical_cost=("clinical_cost", "mean"),
    )
    stress_summary.to_csv(paths["tables"] / "table_18_source_trust_stress_test_summary.csv", index=False)

    severity_perf = primary.groupby(["target_dataset", "feature_set", "conformal_method"], as_index=False).agg(
        mean_coverage=("marginal_coverage", "mean"),
        mean_set_size=("average_prediction_set_size", "mean"),
        mean_referral_rate=("referral_rate", "mean"),
        mean_clinical_cost=("clinical_cost", "mean"),
        mean_class_0_coverage=("class_0_conditional_coverage", "mean"),
        mean_class_1_coverage=("class_1_conditional_coverage", "mean"),
        mean_class_gap=("class_coverage_gap_abs", "mean"),
        mean_source_weight=("selected_source_weight", "mean"),
    )
    severity = severity_perf.merge(
        shift_source_trust,
        on=["target_dataset", "feature_set"],
        how="left",
    )
    severity = severity.loc[severity["conformal_method"].isin([
        "shift_aware_source_trust_atcp",
        "safety_gated_shift_atcp",
        "dp_shift_atcp",
        "multi_objective_dp_shift_atcp",
    ])].copy()
    severity.to_csv(paths["tables"] / "table_19_external_shift_severity_ranking.csv", index=False)

    plot_ablation(ablation, paths, args.dpi)
    plot_shift_aware(conformal, paths, args.dpi)
    plot_shift_atcp_frontier(conformal, paths, args.dpi)
    plot_source_trust_stress(source_trust_stress, paths, args.dpi)
    plot_clinical_tradeoff(tradeoff, paths, args.dpi)

    report = [
        "Publication-ready reviewer-response experiment completed.",
        "",
        f"Output folder: {paths['root'].resolve()}",
        f"Random seed: {args.seed}",
        f"Equal-budget tuning candidates per model family: {args.tuning_candidates}",
        f"Calibration-size ablation repeats: {args.ablation_repeats}",
        f"Bootstrap repeats for primary coverage CI: {args.bootstrap_repeats}",
        "",
        "Key corrections implemented:",
        "- Source calibration method selection uses source calibration fit/select splits only.",
        "- Adaptive calibration method selection uses target calibration fit/select splits only.",
        "- Final external metrics are evaluated only on target test splits.",
        "- Shift-aware baselines include label-shift adjustment, target-only conformal, and importance-weighted source conformal.",
        "- ATCP is recomputed on a consistent target-selected calibration scale before mixing source and target scores.",
        "- Added weighted target-dominant ATCP over source weights 0.00, 0.02, 0.05, 0.10, 0.20, 0.50, and 1.00.",
        "- Added class-conditional weighted ATCP and safety-gated ATCP with target-only fallback.",
        "- Added SHIFT-ATCP with PSI, domain-classifier AUC, prevalence gap, and target-size based source trust.",
        "- Added risk-asymmetric diabetes-protected SHIFT-ATCP.",
        "- Added DP-SHIFT-ATCP with target-selected class-specific coverage pairs for diabetes protection.",
        "- Added multi-objective DP-SHIFT-ATCP using coverage error, diabetes undercoverage, class gap, set size, and clinical cost.",
        "- Added multi-objective safety-gated SHIFT-ATCP using coverage, set size, ECE, class gap, and clinical cost.",
        "- Added class-wise coverage, referral metrics, clinical utility cost, frontier tables, and bootstrap coverage CIs.",
        "- Added source-trust stress testing over simulated target-mix fractions 0%, 25%, 50%, 75%, and 100%.",
        "- Added final Clinical Reliability Trade-off table and figure for SHIFT-ATCP variants.",
        "- Calibration-size ablation varies labeled target calibration size.",
        "- Hyperparameters use equal-budget source-validation candidate selection.",
    ]
    (paths["reports"] / "run_summary.txt").write_text("\n".join(report), encoding="utf-8")
    manifest = {
        "seed": args.seed,
        "input_dir": str(input_dir.resolve()),
        "outdir": str(paths["root"].resolve()),
        "target_splits": split_rows,
        "shift_aware_baselines": [
            "label_shift_adjusted",
            "target_only_conformal",
            "importance_weighted_source_conformal",
            "adaptive_transport_conformal",
            "weighted_target_dominant_atcp",
            "shift_aware_source_trust_atcp",
            "class_conditional_weighted_atcp",
            "risk_asymmetric_shift_atcp",
            "dp_shift_atcp",
            "safety_gated_atcp",
            "safety_gated_shift_atcp",
            "multi_objective_safety_gated_atcp",
            "multi_objective_dp_shift_atcp",
        ],
        "atcp_source_weight_grid": list(ATCP_SOURCE_WEIGHT_GRID),
        "eval_confidence_levels": list(EVAL_CONFIDENCE_LEVELS),
        "risk_asymmetry_delta": RISK_ASYMMETRY_DELTA,
        "dp_shift_atcp": {
            "class_0_confidence_grid": list(DP_CLASS0_CONFIDENCE_GRID),
            "class_1_confidence_grid": list(DP_CLASS1_CONFIDENCE_GRID),
            "selection_score": "5*abs(marginal-target)+2*diabetes_undercoverage+1.5*class_gap+0.8*set_size+0.5*clinical_cost",
        },
        "source_trust_stress_test": {
            "target_mix_grid": list(SOURCE_TRUST_MIX_GRID),
            "description": "Simulated intermediate shift by mixing source and target calibration/test rows.",
        },
        "bootstrap_repeats": args.bootstrap_repeats,
        "shift_atcp": {
            "source_trust_inputs": ["mean_psi", "domain_classifier_auc", "target_calibration_n", "prevalence_gap"],
            "source_weight_formula": "exp(-2*mean_psi)*exp(-3*max(domain_auc-0.5,0))*target_n^-0.5*exp(-2*prevalence_gap)",
        },
        "leakage_correction": {
            "source_calibration_selection": "source calibration fit/select only",
            "adaptive_calibration_selection": "target calibration fit/select only",
            "final_evaluation": "target test only",
        },
    }
    (paths["reports"] / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run(parse_args())
