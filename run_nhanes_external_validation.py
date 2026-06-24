#!/usr/bin/env python3
"""External transportability evaluation on harmonized NHANES 2017--2018 data.

The reduced model uses only sex, age, BMI, HbA1c, and fasting glucose. It is
tuned, calibrated, thresholded, and conformally calibrated solely on the source
dataset before being applied unchanged to NHANES.
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
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

matplotlib_cache = Path(tempfile.gettempdir()) / "diabetes_conformal_matplotlib"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

from src.config import BOOTSTRAP_CONFIDENCE, CLINICAL_TARGET_SENSITIVITIES, CONFIDENCE_LEVELS, MODEL_ORDER
from src.conformal import (
    bootstrap_conformal_confidence_intervals,
    class_probability_matrix,
    evaluate_conformal,
    evaluate_mondrian_conformal,
    prediction_preview,
)
from src.data import clean_dataset, split_train_calib_test
from src.evaluation import (
    bootstrap_point_metric_confidence_intervals,
    classification_report_text,
    evaluate_binary_classifier,
    threshold_analysis,
)
from src.models import (
    calibrate_xgboost_on_training_data,
    oof_calibrated_xgboost_probabilities,
    resolve_xgboost_device,
    tune_models,
)
from src.nhanes import HARMONIZED_FEATURES, build_nhanes_2017_2018, build_source_harmonized
from src.plots import (
    conformal_figures,
    conformal_method_comparison,
    point_prediction_figures,
    set_plot_style,
    subgroup_coverage,
    threshold_tradeoff,
)
from src.subgroups import subgroup_conformal_performance, subgroup_point_performance
from src.utils import (
    base_manifest,
    configure_logging,
    ensure_output_directories,
    list_relative_files,
    save_model,
    sha256_file,
    write_json,
    write_table,
)


def parse_args() -> argparse.Namespace:
    """Parse external validation arguments."""
    parser = argparse.ArgumentParser(description="NHANES external transportability validation")
    parser.add_argument("--source-data", required=True, help="Primary diabetes CSV used to train the reduced model.")
    parser.add_argument("--nhanes-dir", required=True, help="Directory containing DEMO_J, DIQ_J, BMX_J, BPX_J, GHB_J, and GLU_J XPT files.")
    parser.add_argument("--outdir", default="outputs_nhanes_external", help="Separate external-validation output directory.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dpi", default=600, type=int)
    parser.add_argument("--bootstrap-repeats", default=1000, type=int)
    parser.add_argument("--quick", action="store_true", help="Use reduced tuning grids for a smoke test only.")
    return parser.parse_args()


def weighted_external_metrics(y_true: pd.Series, probabilities: np.ndarray, weights: pd.Series) -> dict[str, float]:
    """Calculate survey-weighted external point metrics at threshold 0.5."""
    y = y_true.to_numpy(dtype=int)
    w = weights.to_numpy(dtype=float)
    predictions = (probabilities >= 0.5).astype(int)
    tn = w[(y == 0) & (predictions == 0)].sum()
    fp = w[(y == 0) & (predictions == 1)].sum()
    fn = w[(y == 1) & (predictions == 0)].sum()
    tp = w[(y == 1) & (predictions == 1)].sum()
    specificity = tn / (tn + fp) if tn + fp else np.nan
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    return {
        "weighted_accuracy": float((w * (predictions == y)).sum() / w.sum()),
        "weighted_sensitivity": float(sensitivity), "weighted_specificity": float(specificity),
        "weighted_precision_ppv": float(precision),
        "weighted_roc_auc": float(roc_auc_score(y, probabilities, sample_weight=w)),
        "weighted_pr_auc": float(average_precision_score(y, probabilities, sample_weight=w)),
        "weighted_brier_score": float(np.average((y - probabilities) ** 2, weights=w)),
        "weighted_log_loss": float(log_loss(y, probabilities, sample_weight=w, labels=[0, 1])),
        "weighted_true_negative": float(tn), "weighted_false_positive": float(fp),
        "weighted_false_negative": float(fn), "weighted_true_positive": float(tp),
    }


def main() -> None:
    """Train a harmonized source model and externally evaluate it once on NHANES."""
    args = parse_args()
    source_path = Path(args.source_data).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source data not found: {source_path}")
    paths = ensure_output_directories(Path(args.outdir).expanduser().resolve())
    logger = configure_logging(paths["logs"] / "nhanes_external_validation.log")
    set_plot_style()
    np.random.seed(args.seed)
    logger.info("Loading and cleaning source dataset: %s", source_path)
    source_raw = pd.read_csv(source_path)
    source_cleaned = clean_dataset(source_raw, "diabetes", dataset_profile="generic")
    source = build_source_harmonized(source_cleaned.frame)
    logger.info("Building NHANES 2017-2018 external cohort from %s", args.nhanes_dir)
    nhanes, feature_mapping, cohort_audit = build_nhanes_2017_2018(Path(args.nhanes_dir))
    source.to_csv(paths["data"] / "source_harmonized_reduced_model.csv", index=False)
    nhanes.to_csv(paths["data"] / "nhanes_2017_2018_harmonized_external_cohort.csv", index=False)
    write_table(pd.DataFrame([source_cleaned.audit]), paths["tables"] / "table_nh01_source_cleaning_summary.csv", logger)
    write_table(pd.DataFrame([cohort_audit]), paths["tables"] / "table_nh02_external_cohort_flow.csv", logger)
    write_table(feature_mapping, paths["tables"] / "table_nh03_feature_harmonization.csv", logger)

    source_splits = split_train_calib_test(source, "diabetes", args.seed)
    X_train, X_calib, X_source_test, y_train, y_calib, y_source_test = source_splits
    for name, X, y in (("train", X_train, y_train), ("calibration", X_calib, y_calib), ("source_internal_test", X_source_test, y_source_test)):
        X.to_csv(paths["data"] / f"X_{name}_harmonized.csv", index=False)
        y.rename("diabetes").to_frame().to_csv(paths["data"] / f"y_{name}_harmonized.csv", index=False)
    device = resolve_xgboost_device(args.device, logger)
    models, hyperparameters, best_params, actual_device = tune_models(
        X_train, y_train, device, args.seed, args.quick, paths["tables"], paths["models"], "source_harmonized", logger
    )
    write_table(hyperparameters, paths["tables"] / "table_nh04_source_harmonized_hyperparameters.csv", logger)
    logger.info("Calibrating selected source XGBoost within training data only")
    base_xgboost = models["XGBoost"]
    final_model = calibrate_xgboost_on_training_data(base_xgboost, X_train, y_train)
    save_model(final_model, paths["models"] / "xgboost_source_harmonized_calibrated.joblib")
    logger.info("Generating source-only OOF probabilities for operating-threshold selection")
    oof_probabilities = oof_calibrated_xgboost_probabilities(base_xgboost, X_train, y_train, args.seed)[:, 1]

    internal_rows: list[dict[str, Any]] = []
    for name in MODEL_ORDER:
        model = final_model if name == "XGBoost" else models[name]
        metrics, _, predictions = evaluate_binary_classifier(model, X_source_test, y_source_test, name)
        metrics["best_cv_roc_auc"] = float(hyperparameters.loc[hyperparameters["model"] == name, "best_cv_roc_auc"].iloc[0])
        internal_rows.append(metrics)
        (paths["reports"] / f"classification_report_{name.lower().replace(' ', '_')}_source_harmonized.txt").write_text(
            classification_report_text(y_source_test, predictions), encoding="utf-8"
        )
    write_table(pd.DataFrame(internal_rows).sort_values("roc_auc", ascending=False), paths["tables"] / "table_nh05_source_internal_reduced_model_comparison.csv", logger)

    X_external = nhanes[list(HARMONIZED_FEATURES)].copy()
    y_external = nhanes["diabetes"].astype(int)
    external_metrics, external_probabilities, external_predictions = evaluate_binary_classifier(final_model, X_external, y_external, "Calibrated XGBoost")
    write_table(pd.DataFrame([external_metrics]), paths["tables"] / "table_nh06_external_validation_metrics_unweighted.csv", logger)
    weighted = weighted_external_metrics(y_external, external_probabilities, nhanes["fasting_subsample_weight"])
    write_table(pd.DataFrame([weighted]), paths["tables"] / "table_nh07_external_validation_metrics_survey_weighted.csv", logger)
    (paths["reports"] / "classification_report_xgboost_nhanes_external.txt").write_text(
        classification_report_text(y_external, external_predictions), encoding="utf-8"
    )
    point_prediction_figures(y_external, external_probabilities, external_predictions, paths["figures"], paths["tables"], "nhanes_external", args.dpi)

    thresholds = threshold_analysis(y_train, oof_probabilities, y_external, external_probabilities, CLINICAL_TARGET_SENSITIVITIES)
    write_table(thresholds, paths["tables"] / "table_nh08_external_operating_thresholds.csv", logger)
    threshold_tradeoff(thresholds, paths["figures"] / "figure_nh01_external_threshold_tradeoff.png", args.dpi)
    point_intervals = bootstrap_point_metric_confidence_intervals(y_external, external_probabilities, args.bootstrap_repeats, args.seed)
    point_intervals.insert(1, "ci_confidence_level", BOOTSTRAP_CONFIDENCE)
    write_table(point_intervals, paths["tables"] / "table_nh09_external_bootstrap_confidence_intervals.csv", logger)

    calibration_probabilities = class_probability_matrix(final_model, X_calib)
    external_probability_matrix = class_probability_matrix(final_model, X_external)
    standard, standard_sets, _ = evaluate_conformal(calibration_probabilities, y_calib, external_probability_matrix, y_external, CONFIDENCE_LEVELS)
    standard.insert(0, "method", "standard_split_source_calibration")
    mondrian, mondrian_sets, _ = evaluate_mondrian_conformal(calibration_probabilities, y_calib, external_probability_matrix, y_external, CONFIDENCE_LEVELS)
    mondrian.insert(0, "method", "mondrian_split_source_calibration")
    comparison = pd.concat([standard, mondrian], ignore_index=True, sort=False)
    write_table(standard, paths["tables"] / "table_nh10_external_standard_conformal.csv", logger)
    write_table(mondrian, paths["tables"] / "table_nh11_external_mondrian_conformal.csv", logger)
    write_table(comparison, paths["tables"] / "table_nh12_external_conformal_method_comparison.csv", logger)
    conformal_figures(standard, standard_sets, paths["figures"], "nhanes_external", args.dpi)
    conformal_method_comparison(comparison, paths["figures"] / "figure_nh02_external_conformal_method_comparison.png", args.dpi)
    prediction_preview(external_probability_matrix, y_external, mondrian_sets[0.90], nhanes["SEQN"].astype(str)).to_csv(
        paths["tables"] / "table_nh13_external_mondrian_prediction_preview.csv", index=False
    )
    conformal_intervals = pd.concat([
        bootstrap_conformal_confidence_intervals(y_external, standard_sets, "standard_split_source_calibration", args.bootstrap_repeats, args.seed),
        bootstrap_conformal_confidence_intervals(y_external, mondrian_sets, "mondrian_split_source_calibration", args.bootstrap_repeats, args.seed + 10000),
    ], ignore_index=True)
    conformal_intervals.insert(3, "ci_confidence_level", BOOTSTRAP_CONFIDENCE)
    write_table(conformal_intervals, paths["tables"] / "table_nh14_external_conformal_bootstrap_confidence_intervals.csv", logger)
    subgroup_points = subgroup_point_performance(X_external, y_external, external_probabilities)
    subgroup_conformal = pd.concat([
        subgroup_conformal_performance(X_external, y_external, standard_sets, "standard_split_source_calibration"),
        subgroup_conformal_performance(X_external, y_external, mondrian_sets, "mondrian_split_source_calibration"),
    ], ignore_index=True)
    write_table(subgroup_points, paths["tables"] / "table_nh15_external_subgroup_point_performance.csv", logger)
    write_table(subgroup_conformal, paths["tables"] / "table_nh16_external_subgroup_conformal_performance.csv", logger)
    subgroup_coverage(subgroup_conformal, paths["figures"] / "figure_nh03_external_subgroup_conformal_coverage.png", args.dpi)

    manifest = base_manifest()
    manifest.update({
        "analysis": "NHANES 2017-2018 external transportability evaluation of a harmonized reduced model",
        "source_dataset_path": str(source_path), "source_dataset_sha256": sha256_file(source_path),
        "nhanes_directory": str(Path(args.nhanes_dir).expanduser().resolve()), "random_seed": args.seed,
        "harmonized_features": list(HARMONIZED_FEATURES), "source_outcome": "diabetes",
        "nhanes_outcome": cohort_audit["outcome_definition"], "source_cleaned_rows": len(source),
        "external_cohort_rows": len(nhanes), "external_cohort_audit": cohort_audit,
        "source_split_sizes": {"train": len(y_train), "calibration": len(y_calib), "internal_test": len(y_source_test)},
        "best_params": best_params, "device_requested": args.device, "device_used_for_xgboost": actual_device,
        "bootstrap_repeats": args.bootstrap_repeats,
    })
    write_json(manifest, paths["root"] / "reproducibility_manifest.json")
    manifest["all_output_files_generated"] = list_relative_files(paths["root"])
    write_json(manifest, paths["root"] / "reproducibility_manifest.json")
    summary = """NHANES 2017-2018 external transportability evaluation completed.

The model was trained, calibrated, thresholded, and conformally calibrated using the source dataset only. NHANES labels were not used for tuning, calibration, or threshold selection.

This is an external evaluation of a harmonized reduced model with sex, age, BMI, HbA1c, and fasting glucose. It is not direct validation of the full primary model and uses self-reported clinician-diagnosed diabetes in NHANES (DIQ010) as the external outcome.

Key outputs:
- table_nh02_external_cohort_flow.csv
- table_nh03_feature_harmonization.csv
- table_nh06_external_validation_metrics_unweighted.csv
- table_nh07_external_validation_metrics_survey_weighted.csv
- table_nh08_external_operating_thresholds.csv
- table_nh09_external_bootstrap_confidence_intervals.csv
- table_nh12_external_conformal_method_comparison.csv
- table_nh14_external_conformal_bootstrap_confidence_intervals.csv
- figure_06_roc_curve_nhanes_external.png
- figure_08_calibration_curve_nhanes_external.png
- figure_nh02_external_conformal_method_comparison.png
"""
    (paths["reports"] / "run_summary.txt").write_text(summary, encoding="utf-8")
    print("\nDONE: NHANES external transportability validation completed.\n")
    print("Use the external results only as validation of the harmonized reduced model, not of the full primary model.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.getLogger("diabetes_conformal").exception("NHANES external validation failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from exc
