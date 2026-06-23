"""Model registry, XGBoost device handling, and hyperparameter optimisation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import build_preprocessor
from .utils import safe_stem


@dataclass
class ModelSpec:
    """Estimator metadata used to construct and tune one model family."""

    name: str
    estimator: Any
    grid: dict[str, list[Any]]
    scale_numeric: bool


def xgboost_cuda_available() -> bool:
    """Check whether the installed XGBoost build advertises CUDA support."""
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required. Install project dependencies with: python3 -m pip install -r requirements.txt"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "XGBoost is installed but its native library could not load. On macOS, install the OpenMP runtime "
            "with: brew install libomp. Then verify with: python3 -c \"import xgboost; print(xgboost.__version__)\""
        ) from exc
    try:
        return str(xgb.build_info().get("USE_CUDA", "OFF")).upper() in {"ON", "TRUE", "1"}
    except Exception:
        return False


def resolve_xgboost_device(requested: str, logger: logging.Logger) -> str:
    """Resolve auto/cpu/cuda without claiming CUDA where it is unavailable."""
    available = xgboost_cuda_available()
    if requested == "cuda" and not available:
        logger.warning("CUDA was requested but is not available in this XGBoost build; using CPU.")
        return "cpu"
    if requested == "auto":
        return "cuda" if available else "cpu"
    return requested


def _xgb_estimator(device: str, seed: int) -> Any:
    """Create a version-compatible XGBClassifier."""
    import xgboost as xgb
    common = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    major = int(str(xgb.__version__).split(".")[0])
    if major >= 2:
        return xgb.XGBClassifier(**common, device=device)
    return xgb.XGBClassifier(
        **common,
        tree_method="gpu_hist" if device == "cuda" else "hist",
        predictor="gpu_predictor" if device == "cuda" else "auto",
    )


def build_model_registry(device: str, seed: int, quick: bool = False) -> dict[str, ModelSpec]:
    """Construct all estimators and their full or smoke-test search spaces."""
    if quick:
        lr_grid = {"model__C": [1], "model__penalty": ["l2"], "model__solver": ["lbfgs"], "model__class_weight": [None]}
        svm_grid = {"model__C": [1], "model__kernel": ["rbf"], "model__gamma": ["scale"]}
        dt_grid = {"model__max_depth": [5, None], "model__min_samples_split": [2], "model__min_samples_leaf": [1], "model__criterion": ["gini"], "model__class_weight": [None]}
        rf_grid = {"model__n_estimators": [100], "model__max_depth": [None], "model__min_samples_leaf": [1], "model__max_features": ["sqrt"], "model__class_weight": [None]}
        xgb_grid = {"model__n_estimators": [100], "model__max_depth": [3], "model__learning_rate": [0.1], "model__subsample": [0.8], "model__colsample_bytree": [0.8], "model__reg_lambda": [1.0], "model__min_child_weight": [1]}
    else:
        lr_grid = {"model__C": [0.01, 0.1, 1, 10], "model__penalty": ["l2"], "model__solver": ["lbfgs"], "model__class_weight": [None, "balanced"]}
        svm_grid = {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"], "model__gamma": ["scale"]}
        dt_grid = {"model__max_depth": [3, 5, 10, None], "model__min_samples_split": [2, 10, 25], "model__min_samples_leaf": [1, 5, 10], "model__criterion": ["gini", "entropy"], "model__class_weight": [None, "balanced"]}
        rf_grid = {"model__n_estimators": [200, 400], "model__max_depth": [5, 10, None], "model__min_samples_leaf": [1, 5], "model__max_features": ["sqrt", "log2"], "model__class_weight": [None, "balanced"]}
        xgb_grid = {"model__n_estimators": [200, 400, 600], "model__max_depth": [3, 5, 7], "model__learning_rate": [0.03, 0.05, 0.1], "model__subsample": [0.8, 1.0], "model__colsample_bytree": [0.8, 1.0], "model__reg_lambda": [1.0, 5.0], "model__min_child_weight": [1, 5]}
    return {
        "Logistic Regression": ModelSpec("Logistic Regression", LogisticRegression(max_iter=5000, random_state=seed), lr_grid, True),
        "SVM": ModelSpec("SVM", SVC(probability=True, random_state=seed, cache_size=1000), svm_grid, True),
        "Decision Tree": ModelSpec("Decision Tree", DecisionTreeClassifier(random_state=seed), dt_grid, False),
        "Random Forest": ModelSpec("Random Forest", RandomForestClassifier(random_state=seed, n_jobs=-1), rf_grid, False),
        "XGBoost": ModelSpec("XGBoost", _xgb_estimator(device, seed), xgb_grid, False),
    }


def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    device: str,
    seed: int,
    quick: bool,
    tables_dir: Any,
    models_dir: Any,
    experiment: str,
    logger: logging.Logger,
) -> tuple[dict[str, Pipeline], pd.DataFrame, dict[str, Any], str]:
    """Tune all five models using training data only and save CV artefacts."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fitted: dict[str, Pipeline] = {}
    summary: list[dict[str, Any]] = []
    best_params: dict[str, Any] = {}
    actual_device = device
    for name, spec in build_model_registry(device, seed, quick).items():
        logger.info("Tuning %s for %s using stratified five-fold ROC-AUC CV", name, experiment)
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(X_train, spec.scale_numeric)),
            ("model", spec.estimator),
        ])
        search = GridSearchCV(
            estimator=pipeline, param_grid=spec.grid, scoring="roc_auc", cv=cv,
            n_jobs=-1, return_train_score=True, refit=True, error_score="raise",
        )
        try:
            search.fit(X_train, y_train)
        except Exception as exc:
            if name != "XGBoost" or device != "cuda":
                raise
            logger.warning("CUDA XGBoost fit failed (%s); retrying XGBoost on CPU.", exc)
            fallback = build_model_registry("cpu", seed, quick)["XGBoost"]
            pipeline = Pipeline([
                ("preprocess", build_preprocessor(X_train, fallback.scale_numeric)),
                ("model", fallback.estimator),
            ])
            search = GridSearchCV(
                estimator=pipeline, param_grid=fallback.grid, scoring="roc_auc", cv=cv,
                n_jobs=-1, return_train_score=True, refit=True, error_score="raise",
            )
            search.fit(X_train, y_train)
            actual_device = "cpu"
        stem = safe_stem(name)
        pd.DataFrame(search.cv_results_).to_csv(tables_dir / f"cv_results_{stem}_{experiment}.csv", index=False)
        fitted[name] = search.best_estimator_
        from .utils import save_model
        save_model(search.best_estimator_, models_dir / f"{stem}_{experiment}.joblib")
        best_params[name] = search.best_params_
        summary.append({
            "model": name,
            "best_cv_roc_auc": float(search.best_score_),
            "best_parameters_json": json.dumps(search.best_params_, sort_keys=True),
        })
    return fitted, pd.DataFrame(summary), best_params, actual_device


def calibrate_xgboost_on_training_data(base_model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Calibrate XGBoost probabilities using CV within training data only.

    The dedicated held-out calibration split remains untouched until split
    conformal scores are computed, preserving the conformal protocol.
    """
    try:
        calibrated = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=5)
    except TypeError:  # scikit-learn <= 1.1
        calibrated = CalibratedClassifierCV(base_estimator=base_model, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


def oof_calibrated_xgboost_probabilities(
    base_model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, seed: int
) -> Any:
    """Generate leakage-free, calibrated out-of-fold training probabilities.

    These probabilities are used only to choose prespecified clinical operating
    thresholds. They never use the held-out conformal calibration or test data.
    """
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    try:
        estimator = CalibratedClassifierCV(estimator=clone(base_model), method="sigmoid", cv=5)
    except TypeError:  # scikit-learn <= 1.1
        estimator = CalibratedClassifierCV(base_estimator=clone(base_model), method="sigmoid", cv=5)
    probabilities = cross_val_predict(
        estimator, X_train, y_train, cv=outer_cv, method="predict_proba", n_jobs=-1
    )
    return probabilities
