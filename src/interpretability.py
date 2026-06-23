"""Optional SHAP and mandatory permutation-based model interpretation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .preprocessing import transformed_feature_names
from .utils import write_table


def permutation_importance_analysis(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, tables_dir: Path, figures_dir: Path,
    experiment: str, seed: int, dpi: int, logger: logging.Logger,
) -> pd.DataFrame:
    """Compute raw-feature permutation importance on the held-out test set."""
    logger.info("Computing permutation importance for final XGBoost")
    result = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=seed, n_jobs=-1)
    table = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False, ignore_index=True)
    write_table(table, tables_dir / f"table_11_permutation_importance_{experiment}.csv", logger)
    shown = table.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, max(4.5, 0.35 * len(shown) + 1)))
    ax.barh(shown["feature"], shown["importance_mean"], xerr=shown["importance_std"], color="#2874a6")
    ax.set(xlabel="Decrease in ROC-AUC after permutation", title="Permutation importance: final XGBoost")
    fig.tight_layout()
    fig.savefig(figures_dir / f"figure_13_permutation_importance_{experiment}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return table


def shap_analysis(
    model: Any, X_test: pd.DataFrame, tables_dir: Path, figures_dir: Path, reports_dir: Path,
    experiment: str, seed: int, dpi: int, logger: logging.Logger,
) -> bool:
    """Attempt SHAP analysis; write a clear report instead of failing if unavailable."""
    try:
        import shap  # type: ignore
    except ImportError:
        (reports_dir / f"shap_skipped_{experiment}.txt").write_text(
            "SHAP was not installed; permutation importance was used as the main interpretability method.\n",
            encoding="utf-8",
        )
        logger.info("SHAP is not installed; skipping optional SHAP artefacts.")
        return False
    try:
        sample = X_test.sample(n=min(2000, len(X_test)), random_state=seed)
        preprocessor = model.named_steps["preprocess"]
        transformed = preprocessor.transform(sample)
        names = transformed_feature_names(preprocessor)
        estimator = model.named_steps["model"]
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        importance = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
        write_table(importance, tables_dir / f"table_12_shap_importance_{experiment}.csv", logger)
        plt.figure(figsize=(8.5, 6.5))
        shap.summary_plot(values, transformed, feature_names=names, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(figures_dir / f"figure_14_shap_summary_{experiment}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        plt.figure(figsize=(8.0, 6.0))
        shap.summary_plot(values, transformed, feature_names=names, plot_type="bar", show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(figures_dir / f"figure_15_shap_bar_{experiment}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        return True
    except Exception as exc:
        (reports_dir / f"shap_skipped_{experiment}.txt").write_text(
            f"SHAP artefacts were skipped: {exc}\nPermutation importance remains available.\n", encoding="utf-8"
        )
        logger.warning("SHAP artefacts were skipped: %s", exc)
        return False
