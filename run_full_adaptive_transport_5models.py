#!/usr/bin/env python3
"""Adaptive Transport Calibration experiment across Kaggle, NHANES, and Pima.

The source model is trained only on the Kaggle diabetes dataset. NHANES and
Pima are treated as external deployment populations with target-domain
calibration subsets used only for fixed/adaptive recalibration and ATCP.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib_cache = Path(tempfile.gettempdir()) / "adaptive_transport_calibration_mpl"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


TARGET = "diabetes"
CONFIDENCE_LEVELS = (0.80, 0.90, 0.95)
PRIMARY_CONFIDENCE = 0.90
NUMERIC_FEATURES = {
    "age", "bmi", "hypertension", "heart_disease", "glucose", "hba1c",
    "blood_pressure", "insulin", "pregnancies", "diabetes_pedigree",
}
CATEGORICAL_FEATURES = {"sex", "smoking_history", "race", "socioeconomic_status"}
FEATURE_SETS = {
    "model_a_screening": ["age", "sex", "bmi", "hypertension", "heart_disease", "smoking_history"],
    "model_b_routine_clinical": ["age", "sex", "bmi", "hypertension", "heart_disease", "smoking_history", "glucose", "hba1c"],
    "model_c_extended_transport": [
        "age", "sex", "bmi", "hypertension", "heart_disease", "smoking_history",
        "glucose", "hba1c", "blood_pressure", "insulin", "pregnancies", "diabetes_pedigree",
    ],
}
CALIBRATION_METHODS = ("uncalibrated", "platt", "temperature", "isotonic", "beta")


@dataclass
class Paths:
    root: Path
    data: Path
    tables: Path
    figures: Path
    models: Path
    reports: Path
    logs: Path


class ProbabilityCalibrator:
    """Small probability-only calibrator used for Platt, temperature, isotonic, and beta calibration."""

    def __init__(self, method: str):
        self.method = method
        self.model: Any = None
        self.temperature: float = 1.0

    @staticmethod
    def _clip(p: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = ProbabilityCalibrator._clip(p)
        return np.log(p / (1 - p))

    def fit(self, probabilities: np.ndarray, y: Iterable[int]) -> "ProbabilityCalibrator":
        p = self._clip(probabilities)
        y_arr = np.asarray(list(y), dtype=int)
        if self.method == "uncalibrated":
            return self
        if self.method == "platt":
            self.model = LogisticRegression(max_iter=2000)
            self.model.fit(self._logit(p).reshape(-1, 1), y_arr)
            return self
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
            self.model.fit(p, y_arr)
            return self
        if self.method == "temperature":
            logits = self._logit(p)

            def objective(t: float) -> float:
                adjusted = 1 / (1 + np.exp(-(logits / t)))
                return log_loss(y_arr, self._clip(adjusted), labels=[0, 1])

            result = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded")
            self.temperature = float(result.x if result.success else 1.0)
            return self
        if self.method == "beta":
            X = np.column_stack([np.log(p), np.log1p(-p)])
            self.model = LogisticRegression(max_iter=2000)
            self.model.fit(X, y_arr)
            return self
        raise ValueError(f"Unknown calibration method: {self.method}")

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        p = self._clip(probabilities)
        if self.method == "uncalibrated":
            return p
        if self.method == "platt":
            return self._clip(self.model.predict_proba(self._logit(p).reshape(-1, 1))[:, 1])
        if self.method == "isotonic":
            return self._clip(self.model.transform(p))
        if self.method == "temperature":
            return self._clip(1 / (1 + np.exp(-(self._logit(p) / self.temperature))))
        if self.method == "beta":
            X = np.column_stack([np.log(p), np.log1p(-p)])
            return self._clip(self.model.predict_proba(X)[:, 1])
        raise ValueError(f"Unknown calibration method: {self.method}")


class CalibratedProbabilityClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper exposing predict_proba for a fitted model plus probability calibrator."""

    def __init__(self, base_model: Any, calibrator: ProbabilityCalibrator):
        self.base_model = base_model
        self.calibrator = calibrator
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = positive_probabilities(self.base_model, X)
        p = self.calibrator.transform(raw)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Full Adaptive Transport Calibration diabetes experiment with all five model families")
    parser.add_argument("--source-data", default=str(project_root / "diabetes_prediction_dataset.csv"))
    parser.add_argument("--pima-data", default=str(project_root / "diabetes.csv"))
    parser.add_argument("--nhanes-dir", default=str(project_root / "NHANES"))
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "outputs_full_5models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=500)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--quick", action="store_true", help="Use smaller source sample and lighter estimators.")
    parser.add_argument("--max-source-rows", type=int, default=None, help="Optional stratified source subsample.")
    parser.add_argument("--max-plot-rows", type=int, default=3000)
    return parser.parse_args()


def ensure_paths(outdir: Path) -> Paths:
    root = outdir.resolve()
    paths = Paths(
        root=root,
        data=root / "data",
        tables=root / "tables",
        figures=root / "figures",
        models=root / "models",
        reports=root / "reports",
        logs=root / "logs",
    )
    for path in paths.__dict__.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def configure_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("adaptive_transport")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 16,
        "figure.titlesize": 24,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "lines.linewidth": 2.5,
        "savefig.bbox": "tight",
    })


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric = [c for c in X.columns if c in NUMERIC_FEATURES]
    categorical = [c for c in X.columns if c in CATEGORICAL_FEATURES]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
        if scale_numeric:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", one_hot_encoder()),
        ]), categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_pipeline(estimator: Any, X: pd.DataFrame, scale_numeric: bool) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(X, scale_numeric)), ("model", estimator)])


def normalize_binary_target(values: pd.Series) -> pd.Series:
    unique = sorted(pd.Series(values).dropna().unique())
    if len(unique) != 2:
        raise ValueError(f"Expected binary target, found {unique}")
    return values.map({unique[0]: 0, unique[1]: 1}).astype(int)


def clean_source(path: Path, seed: int, max_rows: int | None) -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates().copy()
    if "gender" in df.columns:
        df = df.loc[~df["gender"].astype(str).str.casefold().eq("other")].copy()
    out = pd.DataFrame({
        "dataset": "Kaggle",
        "age": df["age"],
        "sex": df["gender"].replace({"Female": "Female", "Male": "Male"}).fillna("Unknown"),
        "bmi": df["bmi"],
        "hypertension": df["hypertension"],
        "heart_disease": df["heart_disease"],
        "smoking_history": df["smoking_history"].fillna("Unknown"),
        "glucose": df["blood_glucose_level"],
        "hba1c": df["HbA1c_level"],
        "blood_pressure": np.nan,
        "insulin": np.nan,
        "pregnancies": np.nan,
        "diabetes_pedigree": np.nan,
        "race": "Unknown",
        "socioeconomic_status": "Unknown",
        TARGET: normalize_binary_target(df["diabetes"]),
    })
    if max_rows and len(out) > max_rows:
        _, out = train_test_split(out, test_size=max_rows, stratify=out[TARGET], random_state=seed)
    return out.reset_index(drop=True)


def clean_pima(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates().copy()
    for col in ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"):
        df.loc[df[col].eq(0), col] = np.nan
    blood_pressure = df["BloodPressure"]
    out = pd.DataFrame({
        "dataset": "Pima",
        "age": df["Age"],
        "sex": "Unknown",
        "bmi": df["BMI"],
        "hypertension": np.where(blood_pressure.notna(), (blood_pressure >= 80).astype(float), np.nan),
        "heart_disease": np.nan,
        "smoking_history": "Unknown",
        "glucose": df["Glucose"],
        "hba1c": np.nan,
        "blood_pressure": blood_pressure,
        "insulin": df["Insulin"],
        "pregnancies": df["Pregnancies"],
        "diabetes_pedigree": df["DiabetesPedigreeFunction"],
        "race": "Pima/Native American",
        "socioeconomic_status": "Unknown",
        TARGET: normalize_binary_target(df["Outcome"]),
    })
    return out.reset_index(drop=True)


def _find_nhanes_file(directory: Path, filename: str) -> Path:
    matches = [p for p in directory.iterdir() if p.name.casefold() == filename.casefold()]
    if not matches:
        raise FileNotFoundError(f"Missing NHANES file {filename} in {directory}")
    return matches[0]


def _read_xpt(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_sas(path, format="xport")
    keep = [c for c in columns if c in frame.columns]
    missing = set(columns) - set(keep)
    if missing:
        warnings.warn(f"{path.name} lacks optional columns {sorted(missing)}")
    frame = frame[keep].copy()
    frame["SEQN"] = frame["SEQN"].astype("Int64")
    return frame


def clean_nhanes(nhanes_dir: Path) -> pd.DataFrame:
    demo = _read_xpt(_find_nhanes_file(nhanes_dir, "DEMO_J.xpt"), ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDEXPRG", "RIDRETH3", "INDFMPIR"])
    diq = _read_xpt(_find_nhanes_file(nhanes_dir, "DIQ_J.xpt"), ["SEQN", "DIQ010"])
    bmx = _read_xpt(_find_nhanes_file(nhanes_dir, "BMX_J.xpt"), ["SEQN", "BMXBMI"])
    bpx = _read_xpt(_find_nhanes_file(nhanes_dir, "BPX_J.xpt"), ["SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1", "BPXDI2", "BPXDI3"])
    ghb = _read_xpt(_find_nhanes_file(nhanes_dir, "GHB_J.xpt"), ["SEQN", "LBXGH"])
    glu = _read_xpt(_find_nhanes_file(nhanes_dir, "GLU_J.xpt"), ["SEQN", "LBXGLU"])
    df = demo.merge(diq, on="SEQN").merge(bmx, on="SEQN").merge(bpx, on="SEQN", how="left").merge(ghb, on="SEQN").merge(glu, on="SEQN")
    df = df.loc[df["RIDAGEYR"] >= 20].copy()
    if "RIDEXPRG" in df:
        df = df.loc[~df["RIDEXPRG"].eq(1)].copy()
    df = df.loc[df["DIQ010"].isin([1, 2])].copy()
    systolic = df[["BPXSY1", "BPXSY2", "BPXSY3"]].mean(axis=1)
    diastolic = df[["BPXDI1", "BPXDI2", "BPXDI3"]].mean(axis=1)
    race_map = {
        1.0: "Mexican American", 2.0: "Other Hispanic", 3.0: "Non-Hispanic White",
        4.0: "Non-Hispanic Black", 6.0: "Non-Hispanic Asian", 7.0: "Other/Multi-racial",
    }
    pir = df["INDFMPIR"] if "INDFMPIR" in df else pd.Series(np.nan, index=df.index)
    out = pd.DataFrame({
        "dataset": "NHANES",
        "age": df["RIDAGEYR"],
        "sex": df["RIAGENDR"].map({1.0: "Male", 2.0: "Female"}).fillna("Unknown"),
        "bmi": df["BMXBMI"],
        "hypertension": np.where(systolic.notna() | diastolic.notna(), ((systolic >= 130) | (diastolic >= 80)).astype(float), np.nan),
        "heart_disease": np.nan,
        "smoking_history": "Unknown",
        "glucose": df["LBXGLU"],
        "hba1c": df["LBXGH"],
        "blood_pressure": systolic,
        "insulin": np.nan,
        "pregnancies": np.nan,
        "diabetes_pedigree": np.nan,
        "race": df["RIDRETH3"].map(race_map).fillna("Unknown") if "RIDRETH3" in df else "Unknown",
        "socioeconomic_status": pd.cut(pir, bins=[-np.inf, 1.3, 3.5, np.inf], labels=["Low", "Middle", "High"]).astype(object).fillna("Unknown"),
        TARGET: df["DIQ010"].map({2.0: 0, 1.0: 1}).astype(int),
    })
    required = ["age", "bmi", "glucose", "hba1c", TARGET]
    out = out.loc[out[required].notna().all(axis=1)].reset_index(drop=True)
    return out


def feature_harmonization_table(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for feature in FEATURE_SETS["model_c_extended_transport"] + ["race", "socioeconomic_status"]:
        row = {"harmonized_feature": feature}
        for name, df in datasets.items():
            available = feature in df.columns and not df[feature].isna().all()
            missing_pct = float(df[feature].isna().mean() * 100) if feature in df.columns else 100.0
            row[f"{name}_available"] = available
            row[f"{name}_missing_percent"] = missing_pct
        rows.append(row)
    return pd.DataFrame(rows)


def dataset_summary(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in datasets.items():
        counts = df[TARGET].value_counts().reindex([0, 1], fill_value=0)
        rows.append({
            "dataset": name,
            "n": len(df),
            "class_0": int(counts[0]),
            "class_1": int(counts[1]),
            "diabetes_prevalence_percent": float(df[TARGET].mean() * 100),
            "mean_age": float(df["age"].mean()),
            "mean_bmi": float(df["bmi"].mean()),
        })
    return pd.DataFrame(rows)


def psi(source: pd.Series, target: pd.Series, bins: int = 10) -> float:
    s = source.dropna().astype(float).to_numpy()
    t = target.dropna().astype(float).to_numpy()
    if len(s) < 2 or len(t) < 2:
        return float("nan")
    quantiles = np.unique(np.quantile(s, np.linspace(0, 1, bins + 1)))
    if len(quantiles) <= 2:
        quantiles = np.linspace(np.nanmin(s), np.nanmax(s) + 1e-6, bins + 1)
    s_counts, _ = np.histogram(s, bins=quantiles)
    t_counts, _ = np.histogram(t, bins=quantiles)
    s_prop = np.clip(s_counts / max(s_counts.sum(), 1), 1e-6, None)
    t_prop = np.clip(t_counts / max(t_counts.sum(), 1), 1e-6, None)
    return float(np.sum((t_prop - s_prop) * np.log(t_prop / s_prop)))


def kl_divergence(source: pd.Series, target: pd.Series, bins: int = 20) -> float:
    s = source.dropna().astype(float).to_numpy()
    t = target.dropna().astype(float).to_numpy()
    if len(s) < 2 or len(t) < 2:
        return float("nan")
    lo = min(np.nanmin(s), np.nanmin(t))
    hi = max(np.nanmax(s), np.nanmax(t))
    if lo == hi:
        return 0.0
    s_hist, edges = np.histogram(s, bins=bins, range=(lo, hi), density=True)
    t_hist, _ = np.histogram(t, bins=edges, density=True)
    s_hist = np.clip(s_hist, 1e-8, None)
    t_hist = np.clip(t_hist, 1e-8, None)
    return float(np.sum(t_hist * np.log(t_hist / s_hist)) * (edges[1] - edges[0]))


def shift_metrics(source: pd.DataFrame, target: pd.DataFrame, target_name: str) -> pd.DataFrame:
    rows = []
    for feature in sorted(NUMERIC_FEATURES.intersection(source.columns).intersection(target.columns)):
        s = source[feature]
        t = target[feature]
        rows.append({
            "target_dataset": target_name,
            "feature": feature,
            "source_mean": float(s.mean(skipna=True)),
            "target_mean": float(t.mean(skipna=True)),
            "source_std": float(s.std(skipna=True)),
            "target_std": float(t.std(skipna=True)),
            "psi": psi(s, t),
            "wasserstein_distance": float(wasserstein_distance(s.dropna(), t.dropna())) if s.notna().any() and t.notna().any() else float("nan"),
            "kl_divergence_target_vs_source": kl_divergence(s, t),
        })
    return pd.DataFrame(rows)


def prepare_X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df.reindex(columns=features).copy()
    for col in features:
        if col in CATEGORICAL_FEATURES:
            X[col] = X[col].astype(object).where(X[col].notna(), "Unknown")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def available_algorithms(seed: int, quick: bool) -> dict[str, tuple[Any, bool]]:
    """Return all five required model families, failing clearly if any package is absent."""
    rf_estimators = 120 if quick else 300
    xgb_estimators = 150 if quick else 500
    algorithms: dict[str, tuple[Any, bool]] = {
        "Logistic Regression": (LogisticRegression(max_iter=5000, class_weight="balanced"), True),
        "Random Forest": (RandomForestClassifier(n_estimators=rf_estimators, max_depth=10, min_samples_leaf=5, random_state=seed, n_jobs=1, class_weight="balanced"), False),
    }
    try:
        import xgboost as xgb
    except Exception as exc:
        raise ImportError("XGBoost is required for the full five-model experiment. Install it with: python3 -m pip install xgboost") from exc
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise ImportError("LightGBM is required for the full five-model experiment. Install it with: python3 -m pip install lightgbm") from exc
    try:
        from catboost import CatBoostClassifier
    except Exception as exc:
        raise ImportError("CatBoost is required for the full five-model experiment. Install it with: python3 -m pip install catboost") from exc
    algorithms["XGBoost"] = (xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_estimators=xgb_estimators,
        learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
        random_state=seed, n_jobs=1, tree_method="hist",
    ), False)
    algorithms["LightGBM"] = (lgb.LGBMClassifier(
        n_estimators=500 if not quick else 150, learning_rate=0.05, num_leaves=31,
        random_state=seed, n_jobs=1, class_weight="balanced", verbose=-1,
    ), False)
    algorithms["CatBoost"] = (CatBoostClassifier(
        iterations=500 if not quick else 150, depth=6, learning_rate=0.05,
        random_seed=seed, verbose=False, allow_writing_files=False,
    ), False)
    return algorithms


def positive_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    probs = np.asarray(model.predict_proba(X))
    classes = list(model.classes_)
    return probs[:, classes.index(1)] if 1 in classes else probs[:, -1]


def ece_score(y: Iterable[int], p: np.ndarray, bins: int = 10) -> float:
    y_arr = np.asarray(list(y), dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        if mask.any():
            total += mask.mean() * abs(y_arr[mask].mean() - p[mask].mean())
    return float(total)


def mce_score(y: Iterable[int], p: np.ndarray, bins: int = 10) -> float:
    y_arr = np.asarray(list(y), dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    gaps = []
    for i in range(bins):
        mask = (p >= edges[i]) & (p <= edges[i + 1]) if i == bins - 1 else (p >= edges[i]) & (p < edges[i + 1])
        if mask.any():
            gaps.append(abs(y_arr[mask].mean() - p[mask].mean()))
    return float(max(gaps) if gaps else np.nan)


def calibration_slope_intercept(y: Iterable[int], p: np.ndarray) -> tuple[float, float]:
    y_arr = np.asarray(list(y), dtype=int)
    if len(np.unique(y_arr)) < 2:
        return float("nan"), float("nan")
    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    except TypeError:
        lr = LogisticRegression(penalty="none", solver="lbfgs", max_iter=2000)
    try:
        lr.fit(logits.reshape(-1, 1), y_arr)
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])
    except Exception:
        return float("nan"), float("nan")


def classification_metrics(y: pd.Series, p: np.ndarray, label: dict[str, Any]) -> dict[str, Any]:
    y_arr = y.to_numpy(dtype=int)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arr, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else np.nan
    slope, intercept = calibration_slope_intercept(y_arr, p)
    return {
        **label,
        "n": int(len(y_arr)),
        "roc_auc": float(roc_auc_score(y_arr, p)) if len(np.unique(y_arr)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y_arr, p)) if len(np.unique(y_arr)) == 2 else np.nan,
        "accuracy": float(accuracy_score(y_arr, pred)),
        "precision": float(precision_score(y_arr, pred, zero_division=0)),
        "sensitivity": float(recall_score(y_arr, pred, zero_division=0)),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_arr, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_arr, p)),
        "log_loss": float(log_loss(y_arr, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "ece": ece_score(y_arr, p),
        "mce": mce_score(y_arr, p),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def probability_matrix(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.column_stack([1 - p, p])


def conformal_quantile(probs: np.ndarray, y: pd.Series, confidence: float) -> float:
    labels = y.to_numpy(dtype=int)
    scores = np.sort(1 - probs[np.arange(len(labels)), labels])
    k = math.ceil((len(scores) + 1) * confidence)
    return float(scores[k - 1]) if k <= len(scores) else float("inf")


def conformal_sets(probs: np.ndarray, q_hat: float) -> np.ndarray:
    return (1 - probs) <= q_hat


def conformal_metrics(y: pd.Series, sets: np.ndarray, label: dict[str, Any]) -> dict[str, Any]:
    labels = y.to_numpy(dtype=int)
    covered = sets[np.arange(len(labels)), labels]
    sizes = sets.sum(axis=1)
    row: dict[str, Any] = {
        **label,
        "n": int(len(labels)),
        "marginal_coverage": float(covered.mean()),
        "coverage_gap": float(covered.mean() - label.get("confidence_level", PRIMARY_CONFIDENCE)),
        "average_prediction_set_size": float(sizes.mean()),
        "prediction_set_efficiency": float(1 / sizes.mean()) if sizes.mean() else np.nan,
        "singleton_prediction_rate": float((sizes == 1).mean()),
        "empty_set_rate": float((sizes == 0).mean()),
    }
    for cls in (0, 1):
        mask = labels == cls
        row[f"class_{cls}_conditional_coverage"] = float(covered[mask].mean()) if mask.any() else np.nan
    return row


def subgroup_rows(df: pd.DataFrame, y: pd.Series, p: np.ndarray, sets: np.ndarray, label: dict[str, Any]) -> pd.DataFrame:
    rows = []
    subgroup_defs: dict[str, pd.Series] = {
        "age_group": pd.cut(df["age"], bins=[0, 44, 64, np.inf], labels=["20-44", "45-64", "65+"]).astype(object),
        "sex": df["sex"].astype(object),
        "race": df["race"].astype(object),
        "socioeconomic_status": df["socioeconomic_status"].astype(object),
    }
    y_arr = y.to_numpy(dtype=int)
    for subgroup, values in subgroup_defs.items():
        for level in sorted(values.fillna("Unknown").astype(str).unique()):
            mask = values.fillna("Unknown").astype(str).eq(level).to_numpy()
            if mask.sum() < 20:
                continue
            p_m = p[mask]
            y_m = y_arr[mask]
            sets_m = sets[mask]
            pred = (p_m >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_m, pred, labels=[0, 1]).ravel()
            rows.append({
                **label,
                "subgroup": subgroup,
                "level": level,
                "n": int(mask.sum()),
                "roc_auc": float(roc_auc_score(y_m, p_m)) if len(np.unique(y_m)) == 2 else np.nan,
                "ece": ece_score(y_m, p_m),
                "coverage": float(sets_m[np.arange(len(y_m)), y_m].mean()),
                "prediction_set_size": float(sets_m.sum(axis=1).mean()),
                "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
                "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
                "false_positive_rate": float(fp / (fp + tn)) if fp + tn else np.nan,
                "false_negative_rate": float(fn / (fn + tp)) if fn + tp else np.nan,
            })
    return pd.DataFrame(rows)


def reliability_gap(fairness: pd.DataFrame, label_cols: list[str]) -> pd.DataFrame:
    rows = []
    if fairness.empty:
        return pd.DataFrame(rows)
    for keys, group in fairness.groupby(label_cols, dropna=False):
        key_dict = dict(zip(label_cols, keys if isinstance(keys, tuple) else (keys,)))
        rows.append({
            **key_dict,
            "equal_opportunity_difference": float(group["sensitivity"].max() - group["sensitivity"].min()),
            "calibration_difference": float(group["ece"].max() - group["ece"].min()),
            "coverage_gap_between_groups": float(group["coverage"].max() - group["coverage"].min()),
            "reliability_gap": float(np.nanmax([
                group["sensitivity"].max() - group["sensitivity"].min(),
                group["ece"].max() - group["ece"].min(),
                group["coverage"].max() - group["coverage"].min(),
            ])),
        })
    return pd.DataFrame(rows)


def plot_dataset_shift(datasets: dict[str, pd.DataFrame], paths: Paths, dpi: int) -> None:
    features = ["age", "bmi", "glucose", "hba1c"]
    comparisons = [
        ("NHANES", "figure_01a_feature_histograms_kaggle_vs_nhanes.png"),
        ("Pima", "figure_01b_feature_histograms_kaggle_vs_pima.png"),
    ]
    colors = {"Kaggle": "#2A6FBB", "NHANES": "#D95F02", "Pima": "#1B9E77"}
    for target_name, filename in comparisons:
        if target_name not in datasets:
            continue
        long = pd.concat([
            datasets["Kaggle"].assign(dataset="Kaggle"),
            datasets[target_name].assign(dataset=target_name),
        ], ignore_index=True)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        for ax, feature in zip(axes.ravel(), features):
            plot_data = long.loc[long[feature].notna(), ["dataset", feature]]
            sns.histplot(
                data=plot_data, x=feature, hue="dataset", stat="density", common_norm=False,
                bins=28, ax=ax, element="step", fill=False, linewidth=2.8,
                palette={name: colors[name] for name in plot_data["dataset"].unique()},
            )
            ax.set_title(feature.replace("_", " ").title(), pad=14)
            ax.set_xlabel(feature.replace("_", " ").title())
            ax.set_ylabel("Density")
            ax.tick_params(axis="both", labelsize=17)
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title("")
                for text in legend.get_texts():
                    text.set_fontsize(16)
        fig.suptitle(f"Feature Distributions: Kaggle Source vs {target_name}", y=1.02)
        fig.tight_layout()
        fig.savefig(paths.figures / filename, dpi=dpi)
        plt.close(fig)

    available_targets = [target for target, _ in comparisons if target in datasets]
    if available_targets:
        fig, axes = plt.subplots(len(available_targets), len(features), figsize=(22, 5.4 * len(available_targets)), squeeze=False)
        for row_idx, target_name in enumerate(available_targets):
            long = pd.concat([
                datasets["Kaggle"].assign(dataset="Kaggle"),
                datasets[target_name].assign(dataset=target_name),
            ], ignore_index=True)
            for col_idx, feature in enumerate(features):
                ax = axes[row_idx, col_idx]
                plot_data = long.loc[long[feature].notna(), ["dataset", feature]]
                sns.histplot(
                    data=plot_data,
                    x=feature,
                    hue="dataset",
                    stat="density",
                    common_norm=False,
                    bins=28,
                    ax=ax,
                    element="step",
                    fill=False,
                    linewidth=2.9,
                    palette={name: colors[name] for name in plot_data["dataset"].unique()},
                )
                if row_idx == 0:
                    ax.set_title(feature.replace("_", " ").title(), fontsize=24, pad=14)
                else:
                    ax.set_title("")
                ax.set_xlabel(feature.replace("_", " ").title(), fontsize=21, labelpad=8)
                ax.set_ylabel("")
                ax.tick_params(axis="both", labelsize=17)
                legend = ax.get_legend()
                if legend is not None:
                    legend.set_title("")
                    for text in legend.get_texts():
                        text.set_fontsize(16)
                    legend.get_frame().set_alpha(0.92)
        row_centers = np.linspace(0.73, 0.27, len(available_targets))
        for y_pos, target_name in zip(row_centers, available_targets):
            fig.text(
                0.04,
                y_pos,
                f"Kaggle vs {target_name}",
                rotation=90,
                va="center",
                ha="center",
                fontsize=23,
                fontweight="bold",
            )
        fig.supylabel("Density", fontsize=24, x=0.006)
        fig.tight_layout(h_pad=3.0, w_pad=2.0, rect=(0.075, 0, 1, 1))
        fig.savefig(paths.figures / "figure_01_feature_histograms_combined.png", dpi=dpi)
        plt.close(fig)


def plot_low_dimensional_shift(datasets: dict[str, pd.DataFrame], paths: Paths, seed: int, max_rows: int, dpi: int) -> None:
    features = FEATURE_SETS["model_b_routine_clinical"]
    frames = []
    for name, df in datasets.items():
        sample = df.sample(min(len(df), max_rows // len(datasets)), random_state=seed) if len(df) else df
        frames.append(sample.assign(dataset=name))
    combined = pd.concat(frames, ignore_index=True)
    X = prepare_X(combined, features)
    matrix = build_preprocessor(X, scale_numeric=True).fit_transform(X)
    labels = combined["dataset"].to_numpy()
    pca = PCA(n_components=2, random_state=seed).fit_transform(matrix)
    tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=min(30, max(5, len(combined) // 50))).fit_transform(matrix)
    try:
        import umap
        umap_coords = umap.UMAP(n_components=2, random_state=seed).fit_transform(matrix)
    except Exception:
        umap_coords = None
    panels = [("PCA", pca), ("t-SNE", tsne)] + ([("UMAP", umap_coords)] if umap_coords is not None else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6))
    axes = np.atleast_1d(axes)
    for ax, (name, coords) in zip(axes, panels):
        plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "dataset": labels})
        sns.scatterplot(data=plot_df, x="x", y="y", hue="dataset", s=32, alpha=0.72, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.tick_params(axis="both", labelsize=16)
        ax.legend(title="", fontsize=15, markerscale=1.5)
    fig.suptitle("Dataset Shift Visualization")
    fig.savefig(paths.figures / "figure_02_low_dimensional_shift.png", dpi=dpi)
    plt.close(fig)


def plot_model_comparison(metrics: pd.DataFrame, paths: Paths, dpi: int) -> None:
    internal = metrics.loc[metrics["dataset"].eq("Kaggle_internal_test")].copy()
    if internal.empty:
        return
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(data=internal, x="feature_set", y="roc_auc", hue="model", ax=ax)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Internal Source ROC-AUC by Model and Feature Set")
    ax.set_xlabel("Feature Set")
    ax.set_ylabel("ROC-AUC")
    ax.tick_params(axis="x", rotation=18, labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(title="Model", fontsize=14, title_fontsize=15, ncol=2)
    fig.savefig(paths.figures / "figure_03_internal_model_comparison.png", dpi=dpi)
    plt.close(fig)


def plot_calibration_reliability(y: pd.Series, rows: list[tuple[str, np.ndarray]], path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    y_arr = y.to_numpy(dtype=int)
    bins = np.linspace(0, 1, 11)
    for label, p in rows:
        xs, ys = [], []
        for i in range(10):
            mask = (p >= bins[i]) & (p <= bins[i + 1]) if i == 9 else (p >= bins[i]) & (p < bins[i + 1])
            if mask.any():
                xs.append(float(p[mask].mean()))
                ys.append(float(y_arr[mask].mean()))
        ax.plot(xs, ys, marker="o", markersize=8, label=label)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Diabetes Rate")
    ax.set_title("Reliability Diagram")
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(title="", fontsize=15)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_conformal(comparison: pd.DataFrame, paths: Paths, dpi: int) -> None:
    data = comparison.loc[comparison["confidence_level"].eq(PRIMARY_CONFIDENCE)].copy()
    if data.empty:
        return
    method_labels = {
        "split_conformal_source_calibration": "Source split conformal",
        "adaptive_transport_conformal": "Adaptive transport conformal",
    }
    method_order = list(method_labels.values())
    data["method_label"] = data["conformal_method"].map(method_labels).fillna(data["conformal_method"])

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    palette = {
        "Source split conformal": "#4C78A8",
        "Adaptive transport conformal": "#54A24B",
    }
    sns.boxplot(
        data=data,
        x="marginal_coverage",
        y="target_dataset",
        hue="method_label",
        hue_order=method_order,
        orient="h",
        width=0.58,
        fliersize=0,
        linewidth=2.0,
        palette=palette,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="marginal_coverage",
        y="target_dataset",
        hue="method_label",
        hue_order=method_order,
        orient="h",
        dodge=True,
        size=4.0,
        alpha=0.40,
        linewidth=0,
        palette=palette,
        ax=ax,
    )
    ax.axvline(PRIMARY_CONFIDENCE, color="black", linestyle="--", linewidth=2.4, label="Nominal 90%")
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Marginal Coverage", fontsize=22, labelpad=10)
    ax.set_ylabel("Target Dataset", fontsize=22, labelpad=10)
    ax.tick_params(axis="both", labelsize=19)
    ax.grid(axis="x", color="#D0D0D0", linewidth=1.0, alpha=0.8)
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique and label in [*method_order, "Nominal 90%"]:
            unique[label] = handle
    ax.legend(
        unique.values(),
        unique.keys(),
        title="",
        fontsize=17,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(paths.figures / "figure_04_conformal_coverage.png", dpi=dpi)
    plt.close(fig)


def plot_trs(trs: pd.DataFrame, paths: Paths, dpi: int) -> None:
    if trs.empty:
        return
    scenario_palette = {
        "naive_transfer": "#4C78A8",
        "fixed_recalibration": "#F58518",
        "adaptive_calibration": "#54A24B",
    }
    scenario_labels = {
        "naive_transfer": "Naive",
        "fixed_recalibration": "Fixed",
        "adaptive_calibration": "Adaptive",
    }
    feature_labels = {
        "model_a_screening": "Screening",
        "model_b_routine_clinical": "Routine",
        "model_c_extended_transport": "Extended",
    }

    def prepare_top(target: str, limit: int) -> pd.DataFrame:
        top = trs.loc[trs["target_dataset"].eq(target)].sort_values("transport_reliability_score", ascending=False).head(limit).copy()
        top["rank"] = np.arange(1, len(top) + 1)
        top["scenario_label"] = top["scenario"].map(scenario_labels).fillna(top["scenario"])
        top["feature_label"] = top["feature_set"].map(feature_labels).fillna(
            top["feature_set"].str.replace("model_", "", regex=False).str.replace("_", " ")
        )
        top["calibration_label"] = top["calibration_method"].str.replace("_", " ").str.title()
        top["deployment"] = (
            top["rank"].astype(str) + ". " + top["model"]
            + "\n" + top["feature_label"] + " | " + top["scenario_label"] + " | " + top["calibration_label"]
        )
        return top.sort_values(["transport_reliability_score", "rank"], ascending=[True, False])

    def draw_trs_axis(ax: plt.Axes, top: pd.DataFrame, title: str, show_ylabel: bool = True) -> None:
        colors = top["scenario"].map(scenario_palette).fillna("#7F7F7F")
        ax.barh(top["deployment"], top["transport_reliability_score"], color=colors, edgecolor="#333333", linewidth=0.8)
        xmin = max(0.0, math.floor((float(top["transport_reliability_score"].min()) - 0.035) * 20) / 20)
        xmax = min(1.0, math.ceil((float(top["transport_reliability_score"].max()) + 0.035) * 20) / 20)
        if xmax - xmin < 0.12:
            xmax = min(1.0, xmin + 0.12)
        ax.set_xlim(xmin, xmax)
        if title:
            ax.set_title(title, fontsize=22, pad=12)
        ax.set_xlabel("Transport Reliability Score", fontsize=20, labelpad=9)
        ax.set_ylabel("" if show_ylabel else "")
        ax.tick_params(axis="x", labelsize=17)
        ax.tick_params(axis="y", labelsize=15)
        ax.grid(axis="x", color="#D6D6D6", linewidth=1.0, alpha=0.85)
        ax.grid(axis="y", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for y, value in enumerate(top["transport_reliability_score"]):
            ax.text(
                value + 0.004,
                y,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=15,
                fontweight="bold",
                color="#222222",
            )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="#333333", label=scenario_labels[key])
        for key, color in scenario_palette.items()
    ]
    targets = [target for target in ["NHANES", "Pima"] if target in set(trs["target_dataset"])]
    fig, axes = plt.subplots(1, len(targets), figsize=(10.5 * len(targets), 8.2), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, target in zip(axes, targets):
        draw_trs_axis(ax, prepare_top(target, 8), f"{target}: Top 8 TRS Settings")
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, fontsize=18, frameon=True, bbox_to_anchor=(0.5, 0.985))
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.5)
    fig.savefig(paths.figures / "figure_05_transport_reliability_score.png", dpi=dpi)
    plt.close(fig)

    for target in targets:
        top = prepare_top(target, 10)
        fig, ax = plt.subplots(figsize=(12.8, 8.8))
        draw_trs_axis(ax, top, "")
        ax.legend(handles=legend_handles, loc="upper center", ncol=3, fontsize=17, frameon=True, bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(paths.figures / f"figure_05_{target.lower()}_transport_reliability_score.png", dpi=dpi)
        plt.close(fig)


def bootstrap_ci(y: pd.Series, p: np.ndarray, sets: np.ndarray, repeats: int, seed: int, label: dict[str, Any]) -> pd.DataFrame:
    y_arr = y.to_numpy(dtype=int)
    idx0 = np.flatnonzero(y_arr == 0)
    idx1 = np.flatnonzero(y_arr == 1)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {"roc_auc": [], "brier_score": [], "ece": [], "coverage": []}
    for _ in range(repeats):
        selected = np.concatenate([
            rng.choice(idx0, size=len(idx0), replace=True),
            rng.choice(idx1, size=len(idx1), replace=True),
        ])
        samples["roc_auc"].append(float(roc_auc_score(y_arr[selected], p[selected])))
        samples["brier_score"].append(float(brier_score_loss(y_arr[selected], p[selected])))
        samples["ece"].append(ece_score(y_arr[selected], p[selected]))
        samples["coverage"].append(float(sets[selected][np.arange(len(selected)), y_arr[selected]].mean()))
    rows = []
    for metric, values in samples.items():
        rows.append({
            **label,
            "metric": metric,
            "estimate": float(np.mean(values)),
            "ci_95_lower": float(np.quantile(values, 0.025)),
            "ci_95_upper": float(np.quantile(values, 0.975)),
            "bootstrap_repeats": repeats,
        })
    return pd.DataFrame(rows)


def mcnemar_table(y: pd.Series, p_a: np.ndarray, p_b: np.ndarray, label: dict[str, Any]) -> dict[str, Any]:
    from scipy.stats import binomtest
    y_arr = y.to_numpy(dtype=int)
    a_correct = (p_a >= 0.5).astype(int) == y_arr
    b_correct = (p_b >= 0.5).astype(int) == y_arr
    b01 = int((a_correct & ~b_correct).sum())
    b10 = int((~a_correct & b_correct).sum())
    p_value = float(binomtest(min(b01, b10), n=b01 + b10, p=0.5).pvalue) if b01 + b10 else 1.0
    return {**label, "naive_correct_adaptive_wrong": b01, "naive_wrong_adaptive_correct": b10, "mcnemar_exact_p_value": p_value}


def source_split(source: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_dev, test = train_test_split(source, test_size=0.20, stratify=source[TARGET], random_state=seed)
    train, calib = train_test_split(train_dev, test_size=0.25, stratify=train_dev[TARGET], random_state=seed)
    return train.reset_index(drop=True), calib.reset_index(drop=True), test.reset_index(drop=True)


def external_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    calib, test = train_test_split(df, test_size=0.50, stratify=df[TARGET], random_state=seed)
    return calib.reset_index(drop=True), test.reset_index(drop=True)


def run(args: argparse.Namespace) -> None:
    paths = ensure_paths(Path(args.outdir))
    logger = configure_logging(paths.logs / "adaptive_transport_calibration.log")
    set_plot_style()
    rng_seed = args.seed
    max_source_rows = args.max_source_rows if args.max_source_rows else (25000 if args.quick else None)
    logger.info("Loading datasets")
    source = clean_source(Path(args.source_data), rng_seed, max_source_rows)
    pima = clean_pima(Path(args.pima_data))
    nhanes = clean_nhanes(Path(args.nhanes_dir))
    datasets = {"Kaggle": source, "NHANES": nhanes, "Pima": pima}
    for name, df in datasets.items():
        df.to_csv(paths.data / f"{name.lower()}_harmonized.csv", index=False)
    feature_harmonization_table(datasets).to_csv(paths.tables / "table_01_feature_harmonization_missingness.csv", index=False)
    dataset_summary(datasets).to_csv(paths.tables / "table_02_dataset_summary.csv", index=False)
    shifts = pd.concat([shift_metrics(source, nhanes, "NHANES"), shift_metrics(source, pima, "Pima")], ignore_index=True)
    shifts.to_csv(paths.tables / "table_03_dataset_shift_metrics.csv", index=False)
    plot_dataset_shift(datasets, paths, args.dpi)
    plot_low_dimensional_shift(datasets, paths, rng_seed, args.max_plot_rows, args.dpi)

    algorithms = available_algorithms(rng_seed, args.quick)
    train, source_calib, source_test = source_split(source, rng_seed)
    nhanes_calib, nhanes_test = external_split(nhanes, rng_seed)
    pima_calib, pima_test = external_split(pima, rng_seed)
    external = {"NHANES": (nhanes_calib, nhanes_test), "Pima": (pima_calib, pima_test)}

    internal_rows = []
    calibration_rows = []
    scenario_rows = []
    conformal_rows = []
    fairness_frames = []
    bootstrap_frames = []
    mcnemar_rows = []
    best_source_calibrators: dict[tuple[str, str], str] = {}
    scenario_predictions: dict[tuple[str, str, str, str], tuple[pd.Series, np.ndarray, np.ndarray, pd.DataFrame]] = {}

    logger.info("Training source models and evaluating calibration methods")
    for feature_set, features in FEATURE_SETS.items():
        X_train = prepare_X(train, features)
        X_source_calib = prepare_X(source_calib, features)
        X_source_test = prepare_X(source_test, features)
        y_train = train[TARGET]
        y_source_calib = source_calib[TARGET]
        y_source_test = source_test[TARGET]
        for model_name, (estimator, scale_numeric) in algorithms.items():
            logger.info("Fitting %s / %s", feature_set, model_name)
            model = make_pipeline(clone(estimator), X_train, scale_numeric)
            model.fit(X_train, y_train)
            model_path = paths.models / f"{feature_set}_{model_name.lower().replace(' ', '_')}.joblib"
            joblib.dump(model, model_path)
            source_calib_raw = positive_probabilities(model, X_source_calib)
            source_test_raw = positive_probabilities(model, X_source_test)
            internal_rows.append(classification_metrics(y_source_test, source_test_raw, {
                "dataset": "Kaggle_internal_test", "feature_set": feature_set, "model": model_name, "calibration_method": "uncalibrated",
            }))
            fitted_calibrators: dict[str, ProbabilityCalibrator] = {}
            for method in CALIBRATION_METHODS:
                calibrator = ProbabilityCalibrator(method).fit(source_calib_raw, y_source_calib)
                fitted_calibrators[method] = calibrator
                for dataset_name, eval_df in {"Kaggle_internal_test": source_test, "NHANES": nhanes, "Pima": pima}.items():
                    X_eval = prepare_X(eval_df, features)
                    raw = positive_probabilities(model, X_eval)
                    calibrated = calibrator.transform(raw)
                    calibration_rows.append(classification_metrics(eval_df[TARGET], calibrated, {
                        "dataset": dataset_name, "feature_set": feature_set, "model": model_name, "calibration_method": method,
                    }))
            cal_df_temp = pd.DataFrame([r for r in calibration_rows if r["feature_set"] == feature_set and r["model"] == model_name])
            rank_df = cal_df_temp.groupby("calibration_method", as_index=False).agg(mean_ece=("ece", "mean"), mean_brier=("brier_score", "mean"))
            rank_df["rank_score"] = rank_df["mean_ece"].rank() + rank_df["mean_brier"].rank()
            best_method = str(rank_df.sort_values(["rank_score", "mean_ece"]).iloc[0]["calibration_method"])
            best_source_calibrators[(feature_set, model_name)] = best_method

            for target_name, (target_calib, target_test) in external.items():
                X_target_calib = prepare_X(target_calib, features)
                X_target_test = prepare_X(target_test, features)
                y_target_calib = target_calib[TARGET]
                y_target_test = target_test[TARGET]
                target_calib_raw = positive_probabilities(model, X_target_calib)
                target_test_raw = positive_probabilities(model, X_target_test)

                scenario_probabilities: dict[str, tuple[str, np.ndarray, ProbabilityCalibrator | None]] = {}
                scenario_probabilities["naive_transfer"] = ("uncalibrated", target_test_raw, None)

                fixed_cal = ProbabilityCalibrator(best_method).fit(target_calib_raw, y_target_calib)
                scenario_probabilities["fixed_recalibration"] = (best_method, fixed_cal.transform(target_test_raw), fixed_cal)

                target_method_scores = []
                target_calibrators: dict[str, ProbabilityCalibrator] = {}
                for method in CALIBRATION_METHODS:
                    cal = ProbabilityCalibrator(method).fit(target_calib_raw, y_target_calib)
                    target_calibrators[method] = cal
                    p_cal = cal.transform(target_calib_raw)
                    target_method_scores.append({"method": method, "ece": ece_score(y_target_calib, p_cal), "brier": brier_score_loss(y_target_calib, p_cal)})
                target_rank = pd.DataFrame(target_method_scores)
                target_rank["rank_score"] = target_rank["ece"].rank() + target_rank["brier"].rank()
                adaptive_method = str(target_rank.sort_values(["rank_score", "ece"]).iloc[0]["method"])
                adaptive_cal = target_calibrators[adaptive_method]
                scenario_probabilities["adaptive_calibration"] = (adaptive_method, adaptive_cal.transform(target_test_raw), adaptive_cal)

                naive_p = scenario_probabilities["naive_transfer"][1]
                adaptive_p = scenario_probabilities["adaptive_calibration"][1]
                mcnemar_rows.append(mcnemar_table(y_target_test, naive_p, adaptive_p, {
                    "target_dataset": target_name, "feature_set": feature_set, "model": model_name,
                    "comparison": "naive_transfer_vs_adaptive_calibration",
                }))

                source_calib_probs = probability_matrix(fitted_calibrators[best_method].transform(source_calib_raw))
                for scenario, (method, p_test, scenario_calibrator) in scenario_probabilities.items():
                    row = classification_metrics(y_target_test, p_test, {
                        "target_dataset": target_name, "feature_set": feature_set, "model": model_name,
                        "scenario": scenario, "calibration_method": method,
                    })
                    source_internal_auc = internal_rows[-1]["roc_auc"]
                    naive_auc = classification_metrics(y_target_test, naive_p, {})["roc_auc"]
                    row["change_from_internal_auc"] = float(row["roc_auc"] - source_internal_auc)
                    row["change_from_naive_auc"] = float(row["roc_auc"] - naive_auc)
                    scenario_rows.append(row)

                    test_probs = probability_matrix(p_test)
                    for confidence in CONFIDENCE_LEVELS:
                        q_baseline = conformal_quantile(source_calib_probs, y_source_calib, confidence)
                        baseline_sets = conformal_sets(test_probs, q_baseline)
                        conformal_rows.append(conformal_metrics(y_target_test, baseline_sets, {
                            "target_dataset": target_name, "feature_set": feature_set, "model": model_name,
                            "scenario": scenario, "calibration_method": method, "confidence_level": confidence,
                            "conformal_method": "split_conformal_source_calibration",
                        }))
                        target_calib_for_conformal = scenario_calibrator.transform(target_calib_raw) if scenario_calibrator else target_calib_raw
                        combined_probs = np.vstack([source_calib_probs, probability_matrix(target_calib_for_conformal)])
                        combined_y = pd.concat([y_source_calib.reset_index(drop=True), y_target_calib.reset_index(drop=True)], ignore_index=True)
                        q_atcp = conformal_quantile(combined_probs, combined_y, confidence)
                        atcp_sets = conformal_sets(test_probs, q_atcp)
                        conformal_rows.append(conformal_metrics(y_target_test, atcp_sets, {
                            "target_dataset": target_name, "feature_set": feature_set, "model": model_name,
                            "scenario": scenario, "calibration_method": method, "confidence_level": confidence,
                            "conformal_method": "adaptive_transport_conformal",
                        }))
                        if confidence == PRIMARY_CONFIDENCE:
                            fairness_frames.append(subgroup_rows(target_test, y_target_test, p_test, atcp_sets, {
                                "target_dataset": target_name, "feature_set": feature_set, "model": model_name,
                                "scenario": scenario, "calibration_method": method,
                                "conformal_method": "adaptive_transport_conformal",
                            }))
                            bootstrap_frames.append(bootstrap_ci(y_target_test, p_test, atcp_sets, args.bootstrap_repeats, rng_seed, {
                                "target_dataset": target_name, "feature_set": feature_set, "model": model_name,
                                "scenario": scenario, "calibration_method": method,
                            }))
                            scenario_predictions[(target_name, feature_set, model_name, scenario)] = (y_target_test, p_test, atcp_sets, target_test)

                if feature_set == "model_b_routine_clinical" and model_name in {"Logistic Regression", "XGBoost"} and target_name == "NHANES":
                    plot_calibration_reliability(
                        y_target_test,
                        [("naive", naive_p), ("adaptive", adaptive_p)],
                        paths.figures / f"figure_reliability_{model_name.lower().replace(' ', '_')}_{target_name.lower()}.png",
                        args.dpi,
                    )

    internal = pd.DataFrame(internal_rows)
    calibration = pd.DataFrame(calibration_rows)
    scenarios = pd.DataFrame(scenario_rows)
    conformal = pd.DataFrame(conformal_rows)
    fairness = pd.concat(fairness_frames, ignore_index=True) if fairness_frames else pd.DataFrame()
    fair_gaps = reliability_gap(fairness, ["target_dataset", "feature_set", "model", "scenario", "calibration_method", "conformal_method"])
    bootstraps = pd.concat(bootstrap_frames, ignore_index=True) if bootstrap_frames else pd.DataFrame()

    internal.to_csv(paths.tables / "table_04_internal_source_model_metrics.csv", index=False)
    calibration.to_csv(paths.tables / "table_05_calibration_method_metrics_all_datasets.csv", index=False)
    calib_rank = calibration.groupby(["feature_set", "model", "calibration_method"], as_index=False).agg(mean_ece=("ece", "mean"), mean_brier=("brier_score", "mean"))
    calib_rank["rank_score"] = calib_rank.groupby(["feature_set", "model"])["mean_ece"].rank() + calib_rank.groupby(["feature_set", "model"])["mean_brier"].rank()
    calib_rank.sort_values(["feature_set", "model", "rank_score"]).to_csv(paths.tables / "table_06_calibration_method_ranking.csv", index=False)
    scenarios.to_csv(paths.tables / "table_07_external_transportability_scenarios.csv", index=False)
    conformal.to_csv(paths.tables / "table_08_conformal_atcp_comparison.csv", index=False)
    fairness.to_csv(paths.tables / "table_09_fairness_subgroup_metrics.csv", index=False)
    fair_gaps.to_csv(paths.tables / "table_10_fairness_reliability_gaps.csv", index=False)
    bootstraps.to_csv(paths.tables / "table_11_bootstrap_confidence_intervals.csv", index=False)
    pd.DataFrame(mcnemar_rows).to_csv(paths.tables / "table_12_mcnemar_naive_vs_adaptive.csv", index=False)

    mean_shift = shifts.groupby("target_dataset")["psi"].mean().rename("mean_psi").reset_index()
    primary_conf = conformal.loc[
        conformal["confidence_level"].eq(PRIMARY_CONFIDENCE)
        & conformal["conformal_method"].eq("adaptive_transport_conformal")
    ].copy()
    trs = scenarios.merge(primary_conf[[
        "target_dataset", "feature_set", "model", "scenario", "calibration_method",
        "marginal_coverage", "average_prediction_set_size",
    ]], on=["target_dataset", "feature_set", "model", "scenario", "calibration_method"], how="left")
    trs = trs.merge(mean_shift, on="target_dataset", how="left")
    trs = trs.merge(fair_gaps[["target_dataset", "feature_set", "model", "scenario", "calibration_method", "reliability_gap"]], on=["target_dataset", "feature_set", "model", "scenario", "calibration_method"], how="left")
    trs["performance_component"] = trs["roc_auc"].clip(0, 1)
    trs["calibration_component"] = (1 - trs["ece"]).clip(0, 1)
    trs["uncertainty_component"] = (1 - (trs["marginal_coverage"] - PRIMARY_CONFIDENCE).abs() / PRIMARY_CONFIDENCE).clip(0, 1)
    trs["distribution_shift_component"] = (1 - trs["mean_psi"].fillna(0) / 0.25).clip(0, 1)
    trs["fairness_component"] = (1 - trs["reliability_gap"].fillna(0)).clip(0, 1)
    components = ["performance_component", "calibration_component", "uncertainty_component", "distribution_shift_component", "fairness_component"]
    trs["transport_reliability_score"] = trs[components].mean(axis=1)
    trs.sort_values("transport_reliability_score", ascending=False).to_csv(paths.tables / "table_13_transport_reliability_score.csv", index=False)

    plot_model_comparison(internal, paths, args.dpi)
    plot_conformal(conformal, paths, args.dpi)
    plot_trs(trs, paths, args.dpi)

    manifest = {
        "analysis": "Adaptive Transport Calibration: Reliable Clinical Machine Learning Under Dataset Shift",
        "source_data": str(Path(args.source_data).resolve()),
        "pima_data": str(Path(args.pima_data).resolve()),
        "nhanes_dir": str(Path(args.nhanes_dir).resolve()),
        "seed": args.seed,
        "dpi": args.dpi,
        "quick": bool(args.quick),
        "source_rows_used": int(len(source)),
        "datasets": {name: int(len(df)) for name, df in datasets.items()},
        "feature_sets": FEATURE_SETS,
        "calibration_methods": list(CALIBRATION_METHODS),
        "conformal_methods": ["split_conformal_source_calibration", "adaptive_transport_conformal"],
        "models_run": list(algorithms),
        "optional_models_skipped": [],
    }
    (paths.reports / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = f"""Adaptive Transport Calibration experiment completed.

Rows used:
- Kaggle source: {len(source)}
- NHANES external: {len(nhanes)}
- Pima external: {len(pima)}

Outputs are inside: {paths.root}
Figures were saved at {args.dpi} dpi with enlarged publication fonts.

Key tables:
- table_03_dataset_shift_metrics.csv
- table_05_calibration_method_metrics_all_datasets.csv
- table_07_external_transportability_scenarios.csv
- table_08_conformal_atcp_comparison.csv
- table_09_fairness_subgroup_metrics.csv
- table_13_transport_reliability_score.csv
"""
    (paths.reports / "run_summary.txt").write_text(summary, encoding="utf-8")
    logger.info("Completed. Outputs: %s", paths.root)
    print(summary)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
