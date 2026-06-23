"""Descriptive held-out-test subgroup performance and conformal analyses."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .evaluation import threshold_metrics


def _find_column(columns: pd.Index, candidate: str) -> str | None:
    """Find a column case-insensitively."""
    mapping = {str(column).casefold(): str(column) for column in columns}
    return mapping.get(candidate.casefold())


def subgroup_masks(X_test: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """Return prespecified demographic and clinical subgroup masks when available."""
    groups: list[tuple[str, str, np.ndarray]] = []
    gender = _find_column(X_test.columns, "gender") or _find_column(X_test.columns, "sex")
    if gender is not None:
        for value in sorted(pd.unique(X_test[gender].dropna()), key=str):
            groups.append((gender, str(value), X_test[gender].eq(value).to_numpy()))
    age = _find_column(X_test.columns, "age")
    if age is not None and pd.api.types.is_numeric_dtype(X_test[age]):
        age_values = X_test[age]
        if age_values.max() > 20:  # continuous age in years
            bands = pd.cut(age_values, bins=[-np.inf, 29, 44, 59, np.inf], labels=["<30", "30-44", "45-59", "60+"])
            for value in bands.cat.categories:
                groups.append(("age_band", str(value), bands.eq(value).to_numpy()))
    for candidate in ("hypertension", "heart_disease"):
        column = _find_column(X_test.columns, candidate)
        if column is not None:
            for value in sorted(pd.unique(X_test[column].dropna()), key=str):
                groups.append((column, str(value), X_test[column].eq(value).to_numpy()))
    return groups


def subgroup_point_performance(X_test: pd.DataFrame, y_test: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    """Calculate 0.5-threshold test performance for prespecified subgroups."""
    labels = y_test.to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for feature, subgroup, mask in subgroup_masks(X_test):
        if not mask.any():
            continue
        y_group = y_test.iloc[np.flatnonzero(mask)]
        p_group = probabilities[mask]
        row = threshold_metrics(y_group, p_group, 0.5)
        row.update({
            "subgroup_feature": feature, "subgroup": subgroup, "n": int(mask.sum()),
            "class_1_count": int(labels[mask].sum()), "class_1_prevalence_percent": float(labels[mask].mean() * 100),
        })
        if y_group.nunique() == 2:
            row["roc_auc"] = float(roc_auc_score(y_group, p_group))
            row["pr_auc"] = float(average_precision_score(y_group, p_group))
        else:
            row["roc_auc"] = float("nan")
            row["pr_auc"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_conformal_performance(
    X_test: pd.DataFrame, y_test: pd.Series, sets_by_confidence: dict[float, np.ndarray],
    method: str, confidence_levels: tuple[float, ...] = (0.80, 0.90, 0.95),
) -> pd.DataFrame:
    """Calculate subgroup coverage and set efficiency for fixed conformal outputs."""
    labels = y_test.to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for confidence in confidence_levels:
        sets = sets_by_confidence[confidence]
        for feature, subgroup, mask in subgroup_masks(X_test):
            if not mask.any():
                continue
            local_labels = labels[mask]
            local_sets = sets[mask]
            sizes = local_sets.sum(axis=1)
            rows.append({
                "method": method, "confidence_level": confidence, "subgroup_feature": feature,
                "subgroup": subgroup, "n": int(mask.sum()),
                "class_1_count": int(local_labels.sum()),
                "class_1_prevalence_percent": float(local_labels.mean() * 100),
                "empirical_coverage": float(local_sets[np.arange(len(local_labels)), local_labels].mean()),
                "mean_set_size": float(sizes.mean()), "singleton_rate": float((sizes == 1).mean()),
                "doubleton_rate": float((sizes == 2).mean()), "empty_set_rate": float((sizes == 0).mean()),
            })
    return pd.DataFrame(rows)
