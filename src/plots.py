"""Publication-oriented plots saved as 600-dpi PNG artefacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def set_plot_style() -> None:
    """Set consistent manuscript-friendly plotting defaults."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11})


def workflow_diagram(path: Path, dpi: int) -> None:
    """Draw the reproducible analysis workflow using only matplotlib."""
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")
    labels = ["Raw CSV", "Cleaning", "60/20/20 split", "Model tuning", "XGBoost selection", "Calibration", "Conformal prediction", "Evaluation"]
    x_positions = np.linspace(0.06, 0.94, len(labels))
    for index, (x, label) in enumerate(zip(x_positions, labels)):
        ax.text(x, 0.52, label, ha="center", va="center", transform=ax.transAxes,
                bbox={"boxstyle": "round,pad=0.45", "facecolor": "#eaf2f8", "edgecolor": "#2874a6"})
        if index < len(labels) - 1:
            ax.annotate("", xy=(x_positions[index + 1] - 0.045, 0.52), xytext=(x + 0.045, 0.52),
                        xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "color": "#34495e", "lw": 1.4})
    ax.set_title("Reproducible diabetes risk-prediction workflow", pad=14)
    _save(fig, path, dpi)


def class_distribution(table: pd.DataFrame, path: Path, dpi: int) -> None:
    """Plot post-cleaning primary class counts."""
    row = table.iloc[0]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    bars = ax.bar(["0 = Non-diabetic", "1 = Diabetic"], [row["class_0_count"], row["class_1_count"]], color=["#4c78a8", "#e45756"])
    ax.set(ylabel="Records", title="Class distribution after cleaning")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(bar.get_height()):,}", ha="center", va="bottom")
    _save(fig, path, dpi)


def correlation_heatmap(df: pd.DataFrame, target: str, path: Path, dpi: int) -> None:
    """Plot Pearson correlations among numeric variables and target."""
    numeric = df.select_dtypes(include=[np.number]).copy()
    if target in df.columns and target not in numeric.columns:
        numeric[target] = df[target]
    corr = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson correlation heatmap (numeric variables)")
    _save(fig, path, dpi)


def model_comparison(comparison: pd.DataFrame, path: Path, dpi: int) -> None:
    """Plot four complementary test metrics for all tuned models."""
    metrics = ["roc_auc", "pr_auc", "f1_score", "balanced_accuracy"]
    labels = ["ROC-AUC", "PR-AUC", "F1-score", "Balanced accuracy"]
    ordered = comparison.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    positions = np.arange(len(ordered))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 5.0))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    for index, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        ax.bar(positions + (index - 1.5) * width, ordered[metric], width, label=label, color=color)
    ax.set(xticks=positions, xticklabels=ordered["model"], ylim=(0, 1.03), ylabel="Score", title="Test-set model comparison")
    ax.legend(ncol=2, frameon=False)
    _save(fig, path, dpi)


def point_prediction_figures(y_test: pd.Series, probabilities: np.ndarray, predictions: np.ndarray, figures_dir: Path, tables_dir: Path, experiment: str, dpi: int) -> None:
    """Save confusion, ROC, PR, calibration, confidence plots and their curve data."""
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=ax,
                xticklabels=["Non-diabetic", "Diabetic"], yticklabels=["Non-diabetic", "Diabetic"])
    ax.set(xlabel="Predicted class", ylabel="True class", title="Confusion matrix")
    _save(fig, figures_dir / f"figure_05_confusion_matrix_{experiment}.png", dpi)

    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": thresholds}).to_csv(tables_dir / f"curve_roc_{experiment}.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    ax.plot(fpr, tpr, color="#2874a6", lw=2, label="XGBoost")
    ax.plot([0, 1], [0, 1], "--", color="0.4", label="No discrimination")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02), title="ROC curve")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, figures_dir / f"figure_06_roc_curve_{experiment}.png", dpi)

    precision, recall, thresholds = precision_recall_curve(y_test, probabilities)
    pd.DataFrame({"recall": recall, "precision": precision, "threshold": np.append(thresholds, np.nan)}).to_csv(tables_dir / f"curve_pr_{experiment}.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    ax.plot(recall, precision, color="#d35400", lw=2, label="XGBoost")
    ax.axhline(y_test.mean(), ls="--", color="0.4", label="Observed prevalence")
    ax.set(xlabel="Recall (sensitivity)", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title="Precision-recall curve")
    ax.legend(frameon=False, loc="lower left")
    _save(fig, figures_dir / f"figure_07_precision_recall_curve_{experiment}.png", dpi)

    observed, predicted = calibration_curve(y_test, probabilities, n_bins=10, strategy="quantile")
    pd.DataFrame({"mean_predicted_probability": predicted, "observed_frequency": observed}).to_csv(tables_dir / f"curve_calibration_{experiment}.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    ax.plot(predicted, observed, marker="o", color="#2e8b57", lw=2, label="XGBoost")
    ax.plot([0, 1], [0, 1], "--", color="0.4", label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1), title="Calibration curve")
    ax.legend(frameon=False)
    _save(fig, figures_dir / f"figure_08_calibration_curve_{experiment}.png", dpi)

    confidence = np.maximum(probabilities, 1 - probabilities)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.hist(confidence, bins=25, color="#7f8c8d", edgecolor="white")
    ax.set(xlabel="Point-prediction confidence", ylabel="Test records", title="Prediction confidence distribution")
    _save(fig, figures_dir / f"figure_09_confidence_histogram_{experiment}.png", dpi)


def conformal_figures(coverage: pd.DataFrame, sets: dict[float, np.ndarray], figures_dir: Path, experiment: str, dpi: int) -> None:
    """Plot coverage reliability and distribution of conformal prediction-set sizes."""
    fig, ax = plt.subplots(figsize=(6.3, 4.7))
    ax.plot(coverage["confidence_level"], coverage["empirical_coverage"], marker="o", lw=2, label="Overall coverage")
    ax.plot(coverage["confidence_level"], coverage["class_0_coverage"], marker="s", lw=1.5, label="Class 0 coverage")
    ax.plot(coverage["confidence_level"], coverage["class_1_coverage"], marker="^", lw=1.5, label="Class 1 coverage")
    ax.plot([0.5, 0.95], [0.5, 0.95], "--", color="0.35", label="Nominal coverage")
    ax.set(xlabel="Target confidence", ylabel="Empirical coverage", xlim=(0.48, 0.97), ylim=(0.45, 1.02), title="Conformal coverage versus target confidence")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, figures_dir / f"figure_10_conformal_coverage_vs_confidence_{experiment}.png", dpi)

    selected = [0.80, 0.90, 0.95]
    x = np.arange(3)
    width = 0.22
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for index, confidence in enumerate(selected):
        sizes = sets[confidence].sum(axis=1)
        rates = [(sizes == value).mean() for value in [0, 1, 2]]
        ax.bar(x + (index - 1) * width, rates, width, label=f"{confidence:.0%}")
    ax.set(xticks=x, xticklabels=["Empty", "Singleton", "Doubleton"], ylabel="Proportion of test records", ylim=(0, 1), title="Conformal prediction-set size distribution")
    ax.legend(title="Confidence", frameon=False)
    _save(fig, figures_dir / f"figure_11_conformal_set_size_distribution_{experiment}.png", dpi)


def calibration_size_sensitivity(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    """Plot mean coverage plus one standard deviation across calibration subsamples."""
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    palette = {0.80: "#1f77b4", 0.90: "#ff7f0e", 0.95: "#2ca02c"}
    for confidence, group in summary.groupby("confidence_level"):
        group = group.sort_values("calibration_fraction")
        color = palette[float(confidence)]
        x = group["calibration_fraction"].to_numpy() * 100
        y = group["coverage_mean"].to_numpy()
        std = group["coverage_std"].fillna(0).to_numpy()
        ax.plot(x, y, marker="o", lw=2, color=color, label=f"{confidence:.0%} target")
        ax.fill_between(x, y - std, y + std, color=color, alpha=0.18)
        ax.axhline(confidence, color=color, ls="--", lw=1)
    ax.set(xlabel="Calibration fraction (%)", ylabel="Empirical coverage", ylim=(0.45, 1.02), title="Calibration-size sensitivity")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, path, dpi)


def threshold_tradeoff(thresholds: pd.DataFrame, path: Path, dpi: int) -> None:
    """Plot the sensitivity-specificity trade-off for prespecified operating points."""
    ordered = thresholds.sort_values("threshold", ascending=False)
    labels = ["Default 0.50" if pd.isna(value) else f"Target {value:.0%}" for value in ordered["target_training_sensitivity"]]
    positions = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(positions, ordered["recall_sensitivity"], marker="o", lw=2, label="Test sensitivity")
    ax.plot(positions, ordered["specificity"], marker="s", lw=2, label="Test specificity")
    ax.plot(positions, ordered["precision_ppv"], marker="^", lw=2, label="Test PPV")
    ax.set(xticks=positions, xticklabels=labels, ylim=(0, 1.02), ylabel="Score", title="Clinical operating-threshold trade-off")
    ax.legend(frameon=False, ncol=3, loc="lower center")
    _save(fig, path, dpi)


def conformal_method_comparison(comparison: pd.DataFrame, path: Path, dpi: int) -> None:
    """Compare standard and Mondrian empirical/class-wise coverage."""
    selected = comparison[comparison["confidence_level"].isin([0.80, 0.90, 0.95])].copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    metrics = [("empirical_coverage", "Overall"), ("class_0_coverage", "Class 0"), ("class_1_coverage", "Class 1")]
    positions = np.arange(len(selected))
    width = 0.24
    for index, (metric, label) in enumerate(metrics):
        ax.bar(positions + (index - 1) * width, selected[metric], width, label=label)
    labels = [f"{row.method}\n{row.confidence_level:.0%}" for row in selected.itertuples()]
    ax.set(xticks=positions, xticklabels=labels, ylim=(0, 1.05), ylabel="Coverage", title="Standard versus Mondrian conformal coverage")
    ax.legend(frameon=False, ncol=3)
    _save(fig, path, dpi)


def subgroup_coverage(subgroups: pd.DataFrame, path: Path, dpi: int) -> None:
    """Plot 90% conformal coverage by prespecified subgroup and method."""
    selected = subgroups[subgroups["confidence_level"] == 0.90].copy()
    selected["label"] = selected["subgroup_feature"] + ": " + selected["subgroup"]
    pivoted = selected.pivot(index="label", columns="method", values="empirical_coverage").sort_index()
    fig, ax = plt.subplots(figsize=(8.0, max(4.5, 0.42 * len(pivoted) + 1)))
    pivoted.plot.barh(ax=ax, color=["#2874a6", "#d35400"])
    ax.axvline(0.90, color="0.3", ls="--", label="90% target")
    ax.set(xlabel="Empirical coverage", xlim=(0, 1.02), ylabel="", title="Subgroup conformal coverage at 90% confidence")
    ax.legend(frameon=False)
    _save(fig, path, dpi)
