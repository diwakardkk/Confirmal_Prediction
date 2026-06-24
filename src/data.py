"""Dataset discovery, cleaning, splitting, and descriptive summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


LIKELY_DATASET_NAMES = (
    "diabetes_prediction_dataset.csv",
    "diabetes.csv",
    "Cleaned_dataset.csv",
    "Balanced_dataset.csv",
)
PIMA_COLUMNS = {
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age", "Outcome",
}
PIMA_ZERO_AS_MISSING = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")


@dataclass
class CleanedDataset:
    """Cleaned data plus audit information for the manuscript table."""

    frame: pd.DataFrame
    audit: dict[str, Any]
    target_mapping: dict[str, str]
    dataset_profile: str


def resolve_dataset_profile(df: pd.DataFrame, requested: str) -> str:
    """Resolve an explicit profile or identify the standard Pima schema."""
    if requested not in {"auto", "generic", "pima"}:
        raise ValueError("dataset profile must be one of: auto, generic, pima.")
    if requested == "auto":
        return "pima" if PIMA_COLUMNS.issubset(df.columns) else "generic"
    if requested == "pima" and not PIMA_COLUMNS.issubset(df.columns):
        raise ValueError("The Pima profile requires the standard Pima predictor columns and Outcome target.")
    return requested


def find_dataset(requested: str | None, search_roots: list[Path]) -> Path:
    """Resolve an explicit data path or locate a likely dataset name."""
    if requested:
        path = Path(requested).expanduser()
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f"Dataset not found: {path}")
    for root in search_roots:
        if not root.exists():
            continue
        for name in LIKELY_DATASET_NAMES:
            direct = root / name
            if direct.exists():
                return direct.resolve()
            matches = sorted(root.rglob(name))
            if matches:
                return matches[0].resolve()
    joined = ", ".join(LIKELY_DATASET_NAMES)
    raise FileNotFoundError(f"No dataset was supplied or found. Searched for: {joined}")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV file and reject empty inputs."""
    if path.suffix.lower() != ".csv":
        raise ValueError("This pipeline expects a CSV dataset.")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("The dataset contains no rows.")
    return frame


def _encode_binary_target(values: pd.Series) -> tuple[pd.Series, dict[str, str]]:
    """Safely encode an observed binary target as integer 0/1."""
    if values.isna().any():
        raise ValueError("Target contains missing values; target rows must be corrected before analysis.")
    unique = list(pd.unique(values))
    if len(unique) != 2:
        raise ValueError(f"Target must contain exactly two classes; observed {unique!r}.")
    normalized = {str(value).strip().lower(): value for value in unique}
    positive_names = {"1", "true", "yes", "positive", "diabetes", "diabetic"}
    positive = next((value for key, value in normalized.items() if key in positive_names), None)
    if positive is None:
        try:
            positive = sorted(unique)[-1]
        except TypeError:
            positive = unique[-1]
    negative = next(value for value in unique if value != positive)
    encoded = values.map({negative: 0, positive: 1}).astype(int)
    return encoded, {"0": str(negative), "1": str(positive)}


def clean_dataset(df: pd.DataFrame, target_col: str, dataset_profile: str = "generic") -> CleanedDataset:
    """Remove exact duplicates and ``gender=Other`` while preserving feature missingness."""
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} is not present in the dataset.")
    profile = resolve_dataset_profile(df, dataset_profile)
    original_rows = len(df)
    duplicates = int(df.duplicated().sum())
    cleaned = df.drop_duplicates().copy()
    rows_after_duplicates = len(cleaned)
    gender_other_rows = 0
    if "gender" in cleaned.columns:
        gender_text = cleaned["gender"].astype("string").str.strip().str.casefold()
        other_mask = gender_text.eq("other").fillna(False)
        gender_other_rows = int(other_mask.sum())
        cleaned = cleaned.loc[~other_mask].copy()
    zero_cells_converted = 0
    if profile == "pima":
        for column in PIMA_ZERO_AS_MISSING:
            zero_mask = cleaned[column].eq(0)
            zero_cells_converted += int(zero_mask.sum())
            cleaned.loc[zero_mask, column] = np.nan
    encoded, mapping = _encode_binary_target(cleaned[target_col])
    cleaned[target_col] = encoded
    missing_by_column = cleaned.isna().sum()
    predictors = cleaned.drop(columns=[target_col])
    numeric = predictors.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in predictors.columns if column not in numeric]
    audit = {
        "original_rows": original_rows,
        "duplicate_rows_removed": duplicates,
        "rows_after_duplicate_removal": rows_after_duplicates,
        "gender_other_rows_removed": gender_other_rows,
        "final_rows": len(cleaned),
        "total_features": len(predictors.columns),
        "numeric_features": len(numeric),
        "categorical_features": len(categorical),
        "target_column": target_col,
        "class_0_count": int((cleaned[target_col] == 0).sum()),
        "class_1_count": int((cleaned[target_col] == 1).sum()),
        "diabetes_prevalence_percent": float(cleaned[target_col].mean() * 100),
        "missing_cells_total": int(missing_by_column.sum()),
        "missing_cells_by_column_json": json.dumps({k: int(v) for k, v in missing_by_column.items()}),
        "dataset_profile": profile,
        "zero_encoded_missing_cells_converted": zero_cells_converted,
    }
    return CleanedDataset(cleaned.reset_index(drop=True), audit, mapping, profile)


def create_balanced_dataset(df: pd.DataFrame, target_col: str, seed: int) -> pd.DataFrame:
    """Under-sample the majority class; intended only for secondary sensitivity analysis."""
    counts = df[target_col].value_counts()
    if len(counts) != 2:
        raise ValueError("Balanced sensitivity analysis requires two target classes.")
    n = int(counts.min())
    parts = [group.sample(n=n, random_state=seed) for _, group in df.groupby(target_col, sort=True)]
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_train_calib_test(
    df: pd.DataFrame, target_col: str, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create one stratified 60/20/20 train/calibration/test partition."""
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    if y.value_counts().min() < 10:
        raise ValueError("At least 10 observations per class are required for splitting and cross-validation.")
    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_development, y_development, test_size=0.25, stratify=y_development, random_state=seed
    )
    return X_train, X_calib, X_test, y_train, y_calib, y_test


def class_distribution_table(df: pd.DataFrame, target_col: str, stage: str) -> pd.DataFrame:
    """Build a one-row class distribution table."""
    counts = df[target_col].value_counts().reindex([0, 1], fill_value=0)
    return pd.DataFrame([{
        "dataset_stage": stage,
        "class_0_count": int(counts[0]),
        "class_1_count": int(counts[1]),
        "total": int(len(df)),
        "class_1_prevalence_percent": float(100 * counts[1] / len(df)),
    }])


def split_distribution_table(splits: dict[str, pd.Series]) -> pd.DataFrame:
    """Summarise class composition of named data splits."""
    rows = []
    for name, y in splits.items():
        counts = y.value_counts().reindex([0, 1], fill_value=0)
        rows.append({
            "split": name,
            "total_samples": int(len(y)),
            "class_0_count": int(counts[0]),
            "class_1_count": int(counts[1]),
            "class_1_prevalence_percent": float(100 * counts[1] / len(y)),
        })
    return pd.DataFrame(rows)


def descriptive_statistics_table(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Combine numeric summaries and categorical distributions in one long table."""
    rows: list[dict[str, Any]] = []
    for feature in df.columns:
        if feature == target_col:
            continue
        series = df[feature]
        if pd.api.types.is_numeric_dtype(series):
            summary = series.describe(percentiles=[0.25, 0.5, 0.75])
            rows.append({
                "feature": feature, "variable_type": "numeric", "category": None,
                "count": int(summary["count"]), "mean": float(summary["mean"]),
                "std": float(summary["std"]), "min": float(summary["min"]),
                "q25": float(summary["25%"]), "median": float(summary["50%"]),
                "q75": float(summary["75%"]), "max": float(summary["max"]),
                "percentage": None,
            })
        else:
            counts = series.fillna("<missing>").value_counts(dropna=False)
            for category, count in counts.items():
                rows.append({
                    "feature": feature, "variable_type": "categorical", "category": str(category),
                    "count": int(count), "mean": None, "std": None, "min": None, "q25": None,
                    "median": None, "q75": None, "max": None,
                    "percentage": float(100 * count / len(series)),
                })
    return pd.DataFrame(rows)
