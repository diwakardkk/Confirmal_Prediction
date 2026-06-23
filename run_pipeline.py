#!/usr/bin/env python3
"""One-command reproducible diabetes risk-prediction analysis.

The primary analysis always uses the cleaned original-prevalence dataset.
Balanced under-sampling is optional and is labelled only as a sensitivity analysis.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Keep Matplotlib's font cache writable even on managed systems.
matplotlib_cache = Path(tempfile.gettempdir()) / "diabetes_conformal_matplotlib"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

from src.config import (
    BOOTSTRAP_CONFIDENCE,
    CLINICAL_TARGET_SENSITIVITIES,
    CONFIDENCE_LEVELS,
    MODEL_ORDER,
    SENSITIVITY_CONFIDENCES,
    SENSITIVITY_FRACTIONS,
    RunConfig,
)
from src.conformal import (
    bootstrap_conformal_confidence_intervals,
    calibration_size_sensitivity,
    class_probability_matrix,
    evaluate_conformal,
    evaluate_mondrian_conformal,
    prediction_preview,
)
from src.data import (
    class_distribution_table,
    clean_dataset,
    create_balanced_dataset,
    descriptive_statistics_table,
    find_dataset,
    load_dataset,
    split_distribution_table,
    split_train_calib_test,
)
from src.evaluation import (
    bootstrap_point_metric_confidence_intervals,
    classification_report_text,
    evaluate_binary_classifier,
    threshold_analysis,
)
from src.interpretability import permutation_importance_analysis, shap_analysis
from src.models import (
    calibrate_xgboost_on_training_data,
    oof_calibrated_xgboost_probabilities,
    resolve_xgboost_device,
    tune_models,
)
from src.plots import (
    calibration_size_sensitivity as plot_calibration_size_sensitivity,
    class_distribution,
    conformal_method_comparison,
    conformal_figures,
    correlation_heatmap,
    model_comparison,
    point_prediction_figures,
    set_plot_style,
    subgroup_coverage,
    threshold_tradeoff,
    workflow_diagram,
)
from src.preprocessing import infer_feature_types
from src.subgroups import subgroup_conformal_performance, subgroup_point_performance
from src.utils import (
    base_manifest,
    configure_logging,
    ensure_output_directories,
    list_relative_files,
    sha256_file,
    write_json,
    write_table,
)


def parse_args() -> argparse.Namespace:
    """Parse the command-line interface documented in the README."""
    parser = argparse.ArgumentParser(description="Reproducible diabetes conformal-prediction pipeline")
    parser.add_argument("--data", default=None, help="CSV path. If omitted, likely dataset filenames are searched.")
    parser.add_argument("--outdir", default="outputs", help="Output directory (default: outputs).")
    parser.add_argument("--target", default="diabetes", help="Binary target column (default: diabetes).")
    parser.add_argument("--seed", default=42, type=int, help="Random seed (default: 42).")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="XGBoost device selection.")
    parser.add_argument("--dpi", default=600, type=int, help="PNG resolution (default: 600).")
    parser.add_argument("--bootstrap-repeats", default=1000, type=int, help="Stratified test bootstrap repeats (default: 1000).")
    parser.add_argument("--run-balanced-sensitivity", action="store_true", help="Run a secondary under-sampled balanced analysis.")
    parser.add_argument("--quick", action="store_true", help="Use small tuning grids for a smoke test.")
    parser.add_argument(
        "--experiment-label", default="original_prevalence",
        help="Portable label used in generated filenames (default: original_prevalence).",
    )
    parser.add_argument(
        "--analysis-description", default="cleaned original-prevalence dataset",
        help="Description recorded in the reproducibility manifest and report.",
    )
    return parser.parse_args()


def _save_split_data(paths: dict[str, Path], splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series], target: str, experiment: str) -> None:
    """Save predictors and outcome files for audit/reuse without mixing data splits."""
    X_train, X_calib, X_test, y_train, y_calib, y_test = splits
    for name, X, y in (("train", X_train, y_train), ("calib", X_calib, y_calib), ("test", X_test, y_test)):
        X.to_csv(paths["data"] / f"X_{name}_{experiment}.csv", index=False)
        y.rename(target).to_frame().to_csv(paths["data"] / f"y_{name}_{experiment}.csv", index=False)


def _analysis_summary(
    frame: pd.DataFrame, target: str, metrics: dict[str, Any], coverage: pd.DataFrame, experiment: str
) -> dict[str, Any]:
    """Build one row's contents for primary-versus-balanced comparison."""
    counts = frame[target].value_counts().reindex([0, 1], fill_value=0)
    by_confidence = coverage.set_index("confidence_level")
    return {
        "experiment": experiment,
        "dataset_type": "original prevalence" if experiment == "original_prevalence" else "separate dataset analysis",
        "n_samples": int(len(frame)), "class_0_count": int(counts[0]), "class_1_count": int(counts[1]),
        "xgboost_accuracy": metrics["accuracy"], "xgboost_balanced_accuracy": metrics["balanced_accuracy"],
        "xgboost_recall_sensitivity": metrics["recall_sensitivity"], "xgboost_specificity": metrics["specificity"],
        "xgboost_f1": metrics["f1_score"], "xgboost_roc_auc": metrics["roc_auc"],
        "xgboost_pr_auc": metrics["pr_auc"], "xgboost_brier_score": metrics["brier_score"],
        "coverage_80": by_confidence.loc[0.80, "empirical_coverage"],
        "coverage_90": by_confidence.loc[0.90, "empirical_coverage"],
        "coverage_95": by_confidence.loc[0.95, "empirical_coverage"],
        "mean_set_size_90": by_confidence.loc[0.90, "mean_set_size"],
        "class_0_coverage_90": by_confidence.loc[0.90, "class_0_coverage"],
        "class_1_coverage_90": by_confidence.loc[0.90, "class_1_coverage"],
    }


def run_experiment(
    frame: pd.DataFrame,
    config: RunConfig,
    paths: dict[str, Path],
    experiment: str,
    device: str,
    logger: logging.Logger,
    make_exploratory_figures: bool,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, int]]:
    """Run a complete isolated analysis for one dataset composition."""
    target = config.target
    logger.info("Starting %s analysis with %d rows", experiment, len(frame))
    frame.to_csv(paths["data"] / f"cleaned_{experiment}.csv", index=False)
    distributions = class_distribution_table(frame, target, f"cleaned_{experiment}")
    write_table(distributions, paths["tables"] / f"table_02_class_distribution_{experiment}.csv", logger)
    splits = split_train_calib_test(frame, target, config.seed)
    X_train, X_calib, X_test, y_train, y_calib, y_test = splits
    _save_split_data(paths, splits, target, experiment)
    split_table = split_distribution_table({"train": y_train, "calibration": y_calib, "test": y_test})
    write_table(split_table, paths["tables"] / f"table_03_split_distribution_{experiment}.csv", logger)
    descriptive = descriptive_statistics_table(frame, target)
    write_table(descriptive, paths["tables"] / f"table_04_descriptive_statistics_{experiment}.csv", logger)

    models, hyperparameters, best_params, actual_device = tune_models(
        X_train, y_train, device, config.seed, config.quick, paths["tables"], paths["models"], experiment, logger
    )
    # Probability calibration uses only cross-validation within the training split.
    # The held-out calibration split is reserved exclusively for conformal scores.
    logger.info("Calibrating final XGBoost probabilities using training-only five-fold CV")
    base_xgboost = models["XGBoost"]
    models["XGBoost"] = calibrate_xgboost_on_training_data(base_xgboost, X_train, y_train)
    from src.utils import save_model
    save_model(models["XGBoost"], paths["models"] / f"xgboost_{experiment}.joblib")
    logger.info("Generating training-only out-of-fold calibrated probabilities for clinical threshold selection")
    oof_probability_matrix = oof_calibrated_xgboost_probabilities(base_xgboost, X_train, y_train, config.seed)
    oof_positive_probabilities = oof_probability_matrix[:, 1]
    write_table(hyperparameters, paths["tables"] / f"table_05_best_hyperparameters_{experiment}.csv", logger)
    write_json(best_params, paths["reports"] / f"best_hyperparameters_{experiment}.json")

    comparison_rows: list[dict[str, Any]] = []
    final_probabilities: np.ndarray | None = None
    final_predictions: np.ndarray | None = None
    final_metrics: dict[str, Any] | None = None
    for name in MODEL_ORDER:
        metrics, probabilities, predictions = evaluate_binary_classifier(models[name], X_test, y_test, name)
        metrics["best_cv_roc_auc"] = float(hyperparameters.loc[hyperparameters["model"] == name, "best_cv_roc_auc"].iloc[0])
        comparison_rows.append(metrics)
        stem = name.lower().replace(" ", "_")
        (paths["reports"] / f"classification_report_{stem}_{experiment}.txt").write_text(
            classification_report_text(y_test, predictions), encoding="utf-8"
        )
        if name == "XGBoost":
            final_probabilities, final_predictions, final_metrics = probabilities, predictions, metrics
    assert final_probabilities is not None and final_predictions is not None and final_metrics is not None
    comparison = pd.DataFrame(comparison_rows).sort_values("roc_auc", ascending=False, ignore_index=True)
    write_table(comparison, paths["tables"] / f"table_06_model_comparison_{experiment}.csv", logger)
    write_table(pd.DataFrame([final_metrics]), paths["tables"] / f"table_07_final_xgboost_metrics_{experiment}.csv", logger)

    point_prediction_figures(y_test, final_probabilities, final_predictions, paths["figures"], paths["tables"], experiment, config.dpi)
    if make_exploratory_figures:
        workflow_diagram(paths["figures"] / f"figure_01_workflow_diagram_{experiment}.png", config.dpi)
        class_distribution(distributions, paths["figures"] / f"figure_02_class_distribution_{experiment}.png", config.dpi)
        correlation_heatmap(frame, target, paths["figures"] / f"figure_03_correlation_heatmap_{experiment}.png", config.dpi)
        model_comparison(comparison, paths["figures"] / f"figure_04_model_comparison_{experiment}.png", config.dpi)
    else:
        model_comparison(comparison, paths["figures"] / f"figure_04_model_comparison_{experiment}.png", config.dpi)

    thresholds = threshold_analysis(
        y_train, oof_positive_probabilities, y_test, final_probabilities, CLINICAL_TARGET_SENSITIVITIES
    )
    write_table(thresholds, paths["tables"] / f"table_07b_clinical_threshold_analysis_{experiment}.csv", logger)
    threshold_tradeoff(thresholds, paths["figures"] / f"figure_16_clinical_threshold_tradeoff_{experiment}.png", config.dpi)

    point_intervals = bootstrap_point_metric_confidence_intervals(
        y_test, final_probabilities, config.bootstrap_repeats, config.seed
    )
    point_intervals.insert(1, "ci_confidence_level", BOOTSTRAP_CONFIDENCE)
    write_table(point_intervals, paths["tables"] / f"table_07c_xgboost_bootstrap_confidence_intervals_{experiment}.csv", logger)

    point_subgroups = subgroup_point_performance(X_test, y_test, final_probabilities)
    write_table(point_subgroups, paths["tables"] / f"table_14_subgroup_point_performance_{experiment}.csv", logger)

    # The held-out calibration set is used only to calculate conformal scores/thresholds.
    calibration_probabilities = class_probability_matrix(models["XGBoost"], X_calib)
    test_probability_matrix = class_probability_matrix(models["XGBoost"], X_test)
    coverage, sets_by_confidence, _ = evaluate_conformal(
        calibration_probabilities, y_calib, test_probability_matrix, y_test, CONFIDENCE_LEVELS
    )
    coverage.insert(0, "method", "standard_split")
    write_table(coverage, paths["tables"] / f"table_08_conformal_coverage_{experiment}.csv", logger)
    conformal_figures(coverage, sets_by_confidence, paths["figures"], experiment, config.dpi)
    preview = prediction_preview(test_probability_matrix, y_test, sets_by_confidence[0.90], X_test.index)
    write_table(preview, paths["tables"] / f"table_09_prediction_preview_{experiment}.csv", logger)

    mondrian_coverage, mondrian_sets, _ = evaluate_mondrian_conformal(
        calibration_probabilities, y_calib, test_probability_matrix, y_test, CONFIDENCE_LEVELS
    )
    mondrian_coverage.insert(0, "method", "mondrian_split")
    write_table(mondrian_coverage, paths["tables"] / f"table_08b_mondrian_conformal_coverage_{experiment}.csv", logger)
    method_comparison = pd.concat([coverage, mondrian_coverage], ignore_index=True, sort=False)
    write_table(method_comparison, paths["tables"] / f"table_08c_conformal_method_comparison_{experiment}.csv", logger)
    conformal_method_comparison(method_comparison, paths["figures"] / f"figure_17_conformal_method_comparison_{experiment}.png", config.dpi)
    mondrian_preview = prediction_preview(test_probability_matrix, y_test, mondrian_sets[0.90], X_test.index)
    write_table(mondrian_preview, paths["tables"] / f"table_09b_mondrian_prediction_preview_{experiment}.csv", logger)

    conformal_intervals = pd.concat([
        bootstrap_conformal_confidence_intervals(y_test, sets_by_confidence, "standard_split", config.bootstrap_repeats, config.seed),
        bootstrap_conformal_confidence_intervals(y_test, mondrian_sets, "mondrian_split", config.bootstrap_repeats, config.seed + 10000),
    ], ignore_index=True)
    conformal_intervals.insert(3, "ci_confidence_level", BOOTSTRAP_CONFIDENCE)
    write_table(conformal_intervals, paths["tables"] / f"table_08d_conformal_bootstrap_confidence_intervals_{experiment}.csv", logger)

    subgroup_conformal = pd.concat([
        subgroup_conformal_performance(X_test, y_test, sets_by_confidence, "standard_split"),
        subgroup_conformal_performance(X_test, y_test, mondrian_sets, "mondrian_split"),
    ], ignore_index=True)
    write_table(subgroup_conformal, paths["tables"] / f"table_15_subgroup_conformal_performance_{experiment}.csv", logger)
    subgroup_coverage(subgroup_conformal, paths["figures"] / f"figure_18_subgroup_conformal_coverage_{experiment}.png", config.dpi)

    detailed_sensitivity, sensitivity = calibration_size_sensitivity(
        calibration_probabilities, y_calib, test_probability_matrix, y_test,
        SENSITIVITY_FRACTIONS, SENSITIVITY_CONFIDENCES, config.seed, repeats=10,
    )
    detailed_sensitivity.to_csv(paths["tables"] / f"table_10_calibration_size_sensitivity_detailed_{experiment}.csv", index=False)
    write_table(sensitivity, paths["tables"] / f"table_10_calibration_size_sensitivity_{experiment}.csv", logger)
    plot_calibration_size_sensitivity(sensitivity, paths["figures"] / f"figure_12_calibration_size_sensitivity_{experiment}.png", config.dpi)

    permutation_importance_analysis(models["XGBoost"], X_test, y_test, paths["tables"], paths["figures"], experiment, config.seed, config.dpi, logger)
    # SHAP explains the fitted underlying XGBoost model; probability calibration is a post-processing layer.
    shap_analysis(base_xgboost, X_test, paths["tables"], paths["figures"], paths["reports"], experiment, config.seed, config.dpi, logger)
    split_sizes = {"train": len(y_train), "calibration": len(y_calib), "test": len(y_test)}
    return _analysis_summary(frame, target, final_metrics, coverage, experiment), best_params, actual_device, split_sizes


def write_run_summary(paths: dict[str, Path], used_balanced: bool, experiment: str, analysis_description: str) -> None:
    """Write a direct manuscript hand-off note."""
    text = """Diabetes conformal prediction pipeline completed successfully.

Primary analysis used: {description}.
Balanced under-sampling was {balanced}.

Use the generated tables and figures in outputs/tables and outputs/figures for manuscript revision.
Do not use hardcoded numbers, figures, or tables from a previous draft.

Primary manuscript tables:
- table_01_data_cleaning_summary.csv
- table_03_split_distribution_{experiment}.csv
- table_06_model_comparison_{experiment}.csv
- table_07_final_xgboost_metrics_{experiment}.csv
- table_07b_clinical_threshold_analysis_{experiment}.csv
- table_07c_xgboost_bootstrap_confidence_intervals_{experiment}.csv
- table_08_conformal_coverage_{experiment}.csv
- table_08b_mondrian_conformal_coverage_{experiment}.csv
- table_08c_conformal_method_comparison_{experiment}.csv
- table_08d_conformal_bootstrap_confidence_intervals_{experiment}.csv
- table_10_calibration_size_sensitivity_{experiment}.csv
- table_11_permutation_importance_{experiment}.csv
- table_12_shap_importance_{experiment}.csv (if SHAP is available)
- table_14_subgroup_point_performance_{experiment}.csv
- table_15_subgroup_conformal_performance_{experiment}.csv

Primary manuscript figures:
- figure_01_workflow_diagram_{experiment}.png
- figure_02_class_distribution_{experiment}.png
- figure_04_model_comparison_{experiment}.png
- figure_05_confusion_matrix_{experiment}.png
- figure_06_roc_curve_{experiment}.png
- figure_07_precision_recall_curve_{experiment}.png
- figure_08_calibration_curve_{experiment}.png
- figure_10_conformal_coverage_vs_confidence_{experiment}.png
- figure_12_calibration_size_sensitivity_{experiment}.png
- figure_13_permutation_importance_{experiment}.png
- figure_16_clinical_threshold_tradeoff_{experiment}.png
- figure_17_conformal_method_comparison_{experiment}.png
- figure_18_subgroup_conformal_coverage_{experiment}.png
""".format(
        description=analysis_description,
        experiment=experiment,
        balanced="run only as a secondary sensitivity analysis" if used_balanced else "not run",
    )
    (paths["reports"] / "run_summary.txt").write_text(text, encoding="utf-8")


def main() -> None:
    """Run primary analysis and optionally a separate balanced sensitivity analysis."""
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    data_path = find_dataset(args.data, [Path.cwd(), project_dir, project_dir.parent])
    config = RunConfig(
        data_path=data_path, outdir=Path(args.outdir).expanduser().resolve(), target=args.target,
        seed=args.seed, device_requested=args.device, dpi=args.dpi, bootstrap_repeats=args.bootstrap_repeats, quick=args.quick,
        run_balanced_sensitivity=args.run_balanced_sensitivity,
    )
    paths = ensure_output_directories(config.outdir)
    logger = configure_logging(paths["logs"] / "pipeline.log")
    set_plot_style()
    np.random.seed(config.seed)
    logger.info("Loading dataset from %s", config.data_path)
    raw = load_dataset(config.data_path)
    cleaned = clean_dataset(raw, config.target)
    logger.info("Cleaned data: %d rows; %d duplicate rows and %d gender=Other rows removed.", cleaned.audit["final_rows"], cleaned.audit["duplicate_rows_removed"], cleaned.audit["gender_other_rows_removed"])
    write_table(pd.DataFrame([cleaned.audit]), paths["tables"] / "table_01_data_cleaning_summary.csv", logger)
    device = resolve_xgboost_device(config.device_requested, logger)
    experiment_label = args.experiment_label
    primary_summary, primary_best_params, actual_device, primary_split_sizes = run_experiment(
        cleaned.frame, config, paths, experiment_label, device, logger, make_exploratory_figures=True
    )
    summaries = [primary_summary]
    all_best_params: dict[str, Any] = {experiment_label: primary_best_params}
    if config.run_balanced_sensitivity:
        balanced = create_balanced_dataset(cleaned.frame, config.target, config.seed)
        balanced_audit = dict(cleaned.audit)
        balanced_audit.update({"analysis": "secondary balanced under-sampling sensitivity", "final_rows": len(balanced), "class_0_count": int((balanced[config.target] == 0).sum()), "class_1_count": int((balanced[config.target] == 1).sum()), "diabetes_prevalence_percent": float(balanced[config.target].mean() * 100)})
        write_table(pd.DataFrame([balanced_audit]), paths["tables"] / "table_01_data_cleaning_summary_balanced_sensitivity.csv", logger)
        balanced_summary, balanced_best_params, balanced_device, _ = run_experiment(
            balanced, config, paths, "balanced_sensitivity", actual_device, logger, make_exploratory_figures=False
        )
        summaries.append(balanced_summary)
        all_best_params["balanced_sensitivity"] = balanced_best_params
        actual_device = balanced_device
    comparison = pd.DataFrame(summaries)
    if config.run_balanced_sensitivity:
        write_table(comparison, paths["tables"] / "table_13_primary_vs_balanced_sensitivity_summary.csv", logger)

    write_run_summary(paths, config.run_balanced_sensitivity, experiment_label, args.analysis_description)
    numeric_features, categorical_features = infer_feature_types(cleaned.frame.drop(columns=[config.target]))
    manifest = base_manifest()
    manifest.update({
        "random_seed": config.seed, "dataset_path": str(config.data_path), "dataset_hash_sha256": sha256_file(config.data_path),
        "original_rows": int(len(raw)), "cleaned_rows": int(len(cleaned.frame)), "target_column": config.target,
        "target_mapping": cleaned.target_mapping, "feature_columns": cleaned.frame.drop(columns=[config.target]).columns.tolist(),
        "numeric_features": numeric_features, "categorical_features": categorical_features,
        "train_size": primary_split_sizes["train"], "calibration_size": primary_split_sizes["calibration"], "test_size": primary_split_sizes["test"],
        "model_list": list(MODEL_ORDER), "best_params": all_best_params,
        "bootstrap_confidence_level": BOOTSTRAP_CONFIDENCE,
        "bootstrap_repeats": config.bootstrap_repeats,
        "clinical_target_sensitivities": list(CLINICAL_TARGET_SENSITIVITIES),
        "conformal_methods": ["standard_split", "mondrian_split"],
        "device_requested": config.device_requested, "device_used_for_xgboost": actual_device,
        "primary_analysis": args.analysis_description,
        "experiment_label": experiment_label,
        "balanced_analysis": "secondary under-sampled sensitivity" if config.run_balanced_sensitivity else "not run",
    })
    write_json(manifest, paths["root"] / "reproducibility_manifest.json")
    manifest["all_output_files_generated"] = list_relative_files(paths["root"])
    write_json(manifest, paths["root"] / "reproducibility_manifest.json")
    print("\nDONE: Diabetes conformal prediction pipeline completed.\n")
    print("Primary manuscript tables:")
    for name in ("table_01_data_cleaning_summary.csv", f"table_03_split_distribution_{experiment_label}.csv", f"table_06_model_comparison_{experiment_label}.csv", f"table_07_final_xgboost_metrics_{experiment_label}.csv", f"table_07b_clinical_threshold_analysis_{experiment_label}.csv", f"table_07c_xgboost_bootstrap_confidence_intervals_{experiment_label}.csv", f"table_08_conformal_coverage_{experiment_label}.csv", f"table_08b_mondrian_conformal_coverage_{experiment_label}.csv", f"table_08c_conformal_method_comparison_{experiment_label}.csv", f"table_08d_conformal_bootstrap_confidence_intervals_{experiment_label}.csv", f"table_10_calibration_size_sensitivity_{experiment_label}.csv", f"table_11_permutation_importance_{experiment_label}.csv", f"table_12_shap_importance_{experiment_label}.csv if available", f"table_14_subgroup_point_performance_{experiment_label}.csv", f"table_15_subgroup_conformal_performance_{experiment_label}.csv"):
        print(f"- {name}")
    print("\nPrimary manuscript figures:")
    for name in (f"figure_01_workflow_diagram_{experiment_label}.png", f"figure_02_class_distribution_{experiment_label}.png", f"figure_04_model_comparison_{experiment_label}.png", f"figure_05_confusion_matrix_{experiment_label}.png", f"figure_06_roc_curve_{experiment_label}.png", f"figure_07_precision_recall_curve_{experiment_label}.png", f"figure_08_calibration_curve_{experiment_label}.png", f"figure_10_conformal_coverage_vs_confidence_{experiment_label}.png", f"figure_12_calibration_size_sensitivity_{experiment_label}.png", f"figure_13_permutation_importance_{experiment_label}.png", f"figure_16_clinical_threshold_tradeoff_{experiment_label}.png", f"figure_17_conformal_method_comparison_{experiment_label}.png", f"figure_18_subgroup_conformal_coverage_{experiment_label}.png"):
        print(f"- {name}")
    print("\nUse only generated tables and figures for manuscript revision. Do not use old numbers from the previous paper draft.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.getLogger("diabetes_conformal").exception("Pipeline failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc
