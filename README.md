# SHIFT-ATCP: Shift-Aware and Diabetes-Protected Adaptive Conformal Prediction for Reliable Diabetes Classification Under Dataset Shift

This repository contains the code, result tables, figures, and manuscript material for:

**SHIFT-ATCP: Shift-Aware and Diabetes-Protected Adaptive Conformal Prediction for Reliable Diabetes Classification Under Dataset Shift**

The study evaluates whether diabetes prediction models trained on the Kaggle Diabetes Prediction Dataset remain reliable when transported to external target populations. NHANES 2017-2018 is used as the main external validation cohort, and the Pima Indians Diabetes dataset is used as an extreme-shift stress-test cohort.

The main contribution is not only higher classification performance. The repository focuses on external reliability under dataset shift: probability calibration, conformal coverage, diabetes-positive class coverage, referral behavior, and clinical cost.

## Title

SHIFT-ATCP: Shift-Aware and Diabetes-Protected Adaptive Conformal Prediction for Reliable Diabetes Classification Under Dataset Shift

## Abstract

Machine learning models for diabetes prediction often perform well in the development dataset but become less reliable when transported to a new clinical population. This study evaluates external diabetes model transportability from the Kaggle Diabetes Prediction Dataset to NHANES 2017--2018 as the main external cohort and the Pima Indians Diabetes dataset as an extreme-shift stress-test cohort. Five model families were trained on the source cohort: logistic regression, random forest, XGBoost, LightGBM, and CatBoost. Three feature sets were studied: non-invasive screening features, routine clinical features, and extended transport features. To address reviewer concerns about calibration leakage and unclear split-specific metrics, all external datasets were split into target calibration and target test partitions using stratified 50/50 sampling with random seed 42. Calibration method selection used only source calibration or target calibration-select data, and final external metrics were computed only on held-out target test sets. The revised framework introduces SHIFT-ATCP, a shift-aware target-dominant adaptive conformal method that estimates source trust from Population Stability Index (PSI), domain-classifier AUC, target calibration size, and prevalence shift. It also introduces DP-SHIFT-ATCP, a diabetes-protected variant that selects class-specific conformal thresholds to improve diabetes-positive coverage. Dataset shift was severe: mean PSI was 1.545 for NHANES and 1.897 for Pima, and domain-classifier AUC was 1.000 for both targets. The shift-aware source trust mechanism therefore assigned near-zero source weights. Conventional unweighted ATCP undercovered both targets, with mean 90% coverage of 0.8468 on NHANES and 0.7612 on Pima. Safety-gated SHIFT-ATCP restored near-nominal marginal coverage, reaching 0.9031 on NHANES and 0.9059 on Pima. DP-SHIFT-ATCP improved diabetes-positive coverage from 0.6612 to 0.9090 on NHANES and from 0.8092 to 0.9280 on Pima, with larger prediction sets and referral rates. These results show that external diabetes model reliability requires not only discrimination and calibration but also target-aware conformal uncertainty, source-trust estimation, and class-wise clinical safety analysis.

## Current Paper Version

The latest manuscript source is:

```text
outputs_dp_shift_atcp_clinical_novelty/manuscript/access_updated_shift_atcp.tex
```

The separate revision insert package is:

```text
outputs_dp_shift_atcp_clinical_novelty/manuscript_revision_insert/shift_atcp_revision_insert.tex
outputs_dp_shift_atcp_clinical_novelty/shift_atcp_revision_insert_package.zip
```

## Datasets

Three public diabetes-related datasets are used after cleaning and harmonization:

| Dataset | Role | N | Class 0 | Class 1 | Diabetes prevalence |
|---|---:|---:|---:|---:|---:|
| Kaggle Diabetes Prediction Dataset | Source development cohort | 96,128 | 87,646 | 8,482 | 8.82% |
| NHANES 2017-2018 | Main external target cohort | 2,291 | 1,888 | 403 | 17.59% |
| Pima Indians Diabetes | Extreme-shift target cohort | 768 | 500 | 268 | 34.90% |

Harmonized CSV files are saved in:

```text
outputs_dp_shift_atcp_clinical_novelty/data/
```

## Leakage-Safe Splits

All final target results are computed only on held-out target-test partitions. The target-test data are not used for training, hyperparameter tuning, calibration-method selection, source-trust estimation, conformal-threshold selection, or DP-SHIFT operating-point selection.

| Dataset | Split | N | Class 0 | Class 1 | Seed | Stratified by |
|---|---:|---:|---:|---:|---:|---|
| Kaggle | Source train | 57,676 | 52,587 | 5,089 | 42 | Diabetes |
| Kaggle | Source calibration | 19,226 | 17,529 | 1,697 | 42 | Diabetes |
| Kaggle | Source internal test | 19,226 | 17,530 | 1,696 | 42 | Diabetes |
| NHANES | Target calibration | 1,145 | 944 | 201 | 42 | Diabetes |
| NHANES | Target test | 1,146 | 944 | 202 | 42 | Diabetes |
| Pima | Target calibration | 384 | 250 | 134 | 42 | Diabetes |
| Pima | Target test | 384 | 250 | 134 | 42 | Diabetes |

For adaptive calibration and conformal operating-point selection, the target-calibration partition is further divided into target-calibration-fit and target-calibration-select subsets using a 50/50 stratified split with random seed 42.

| Dataset | Target calibration N | Fit N | Fit class 0 | Fit class 1 | Select N | Select class 0 | Select class 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NHANES | 1,145 | 572 | 472 | 100 | 573 | 472 | 101 |
| Pima | 384 | 192 | 125 | 67 | 192 | 125 | 67 |

## Models and Feature Sets

Five model families are evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Three clinically motivated feature settings are used:

- **Model A: Screening features**: non-invasive variables such as age, sex, BMI, hypertension, heart disease, and smoking history.
- **Model B: Routine clinical features**: screening variables plus routine laboratory variables such as blood glucose and HbA1c.
- **Model C: Extended transport features**: all harmonized clinically comparable variables available for transport experiments.

Equal-budget hyperparameter selection is performed only on source validation data. Each model family uses three candidate settings.

## Proposed Reliability Framework

The final paper evaluates conventional transfer and conformal baselines, then introduces two main reliability methods.

**SHIFT-ATCP** estimates source trust from observed source-target shift:

```text
source trust = f(mean PSI, domain-classifier AUC, target calibration size, prevalence gap)
```

When source-target shift is severe, the source weight becomes very small and the conformal threshold becomes target-dominant.

**DP-SHIFT-ATCP** is a diabetes-protected version of SHIFT-ATCP. It uses class-specific conformal thresholds to improve diabetes-positive coverage, because marginal coverage alone can hide poor protection for the minority diabetes-positive class.

The clinical cost values used in the paper are:

```text
false negative cost = 5.0
false positive cost = 1.0
referral cost = 2.0
empty-set cost = 3.0
```

The DP-SHIFT multi-objective score uses fixed weights:

```text
coverage error = 5.0
diabetes undercoverage = 2.0
class coverage gap = 1.5
mean set size = 0.8
clinical cost = 0.5
```

These values are fixed before target-test evaluation.

## Main Results

Severe source-target shift is observed for both external cohorts.

| Target | Mean PSI | Domain AUC | Interpretation |
|---|---:|---:|---|
| NHANES | 1.545 | 1.000 | Severe source-target shift |
| Pima | 1.897 | 1.000 | Extreme source-target shift |

Conventional unweighted ATCP undercovers under shift. Safety-gated SHIFT-ATCP restores near-nominal marginal coverage. DP-SHIFT-ATCP improves diabetes-positive coverage but increases prediction-set size and referral rate.

| Target | Method | Marginal coverage | Diabetes coverage | Mean set size | Referral rate | Clinical cost |
|---|---|---:|---:|---:|---:|---:|
| NHANES | Old ATCP | 0.8468 | 0.5381 | 1.0223 | 0.1059 | 0.6159 |
| NHANES | Safety-gated SHIFT | 0.9031 | 0.6612 | 1.1401 | 0.1582 | 0.6319 |
| NHANES | DP-SHIFT | 0.8852 | 0.9090 | 1.2198 | 0.2198 | 0.6186 |
| NHANES | Risk-asymmetric SHIFT | 0.8983 | 0.9403 | 1.2859 | 0.2859 | 0.7156 |
| Pima | Old ATCP | 0.7612 | 0.4837 | 1.1969 | 0.2488 | 1.3930 |
| Pima | Safety-gated SHIFT | 0.9059 | 0.8092 | 1.4911 | 0.4911 | 1.3427 |
| Pima | DP-SHIFT | 0.8745 | 0.9280 | 1.4750 | 0.4750 | 1.1760 |
| Pima | Risk-asymmetric SHIFT | 0.8933 | 0.9486 | 1.5324 | 0.5324 | 1.2432 |

## Important Result Figure

![Clinical referral and cost profile of SHIFT-ATCP variants](outputs_dp_shift_atcp_clinical_novelty/figures/figure_shift_atcp_clinical_referral_cost.png)

This figure summarizes the clinical referral and cost profile of the main SHIFT-ATCP variants. It shows that diabetes-protected methods increase uncertainty and referral when needed, but improve diabetes-positive coverage and reduce diabetes-positive undercoverage.

The main interpretation is:

- Safety-gated SHIFT-ATCP is best when the goal is near-nominal marginal coverage.
- DP-SHIFT-ATCP is best when the clinical priority is diabetes-positive protection.
- Risk-asymmetric SHIFT gives the highest diabetes-positive coverage but requires more referrals.
- Under severe shift, the estimated source-trust weight is near zero, showing that source calibration should not dominate target conformal thresholds.

## Key Output Files

Latest outputs are stored in:

```text
outputs_dp_shift_atcp_clinical_novelty/
```

Important tables:

```text
outputs_dp_shift_atcp_clinical_novelty/tables/table_01_split_construction.csv
outputs_dp_shift_atcp_clinical_novelty/tables/table_03_dataset_shift_metrics.csv
outputs_dp_shift_atcp_clinical_novelty/tables/table_04_equal_budget_hyperparameter_tuning.csv
outputs_dp_shift_atcp_clinical_novelty/tables/table_08_leakage_safe_external_transportability_scenarios.csv
outputs_dp_shift_atcp_clinical_novelty/tables/table_09_shift_aware_conformal_comparison.csv
outputs_dp_shift_atcp_clinical_novelty/tables/table_10_target_calibration_size_ablation.csv
outputs_dp_shift_atcp_clinical_novelty/tables/table_12_shift_aware_source_trust_metrics.csv
outputs_dp_shift_atcp_clinical_novelty/tables/table_14_clinical_reliability_summary.csv
outputs_dp_shift_atcp_clinical_novelty/tables/table_15_bootstrap_coverage_confidence_intervals.csv
outputs_dp_shift_atcp_clinical_novelty/tables/table_16_diabetes_protected_shift_atcp_selection.csv
outputs_dp_shift_atcp_clinical_novelty/tables/table_18_source_trust_stress_test_summary.csv
outputs_dp_shift_atcp_clinical_novelty/tables/table_20_clinical_reliability_tradeoff_shift_atcp_variants.csv
```

Important figures:

```text
outputs_dp_shift_atcp_clinical_novelty/figures/figure_shift_aware_conformal_baselines.png
outputs_dp_shift_atcp_clinical_novelty/figures/figure_source_trust_stress_test.png
outputs_dp_shift_atcp_clinical_novelty/figures/figure_clinical_reliability_tradeoff_shift_atcp_variants.png
outputs_dp_shift_atcp_clinical_novelty/figures/figure_shift_atcp_clinical_referral_cost.png
outputs_dp_shift_atcp_clinical_novelty/figures/figure_shift_atcp_coverage_efficiency_frontier.png
outputs_dp_shift_atcp_clinical_novelty/figures/figure_target_calibration_size_ablation_coverage.png
outputs_dp_shift_atcp_clinical_novelty/figures/figure_target_calibration_size_ablation_ece.png
```

## Run the Latest Experiment

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the current publication-ready experiment:

```bash
python3 run_publication_ready_review_response.py
```

The script writes all current outputs to:

```text
outputs_dp_shift_atcp_clinical_novelty/
```

## Reproducibility

The latest run uses:

```text
Random seed: 42
Equal-budget tuning candidates per model family: 3
Calibration-size ablation repeats: 20
Bootstrap repeats for primary coverage CI: 300
Main conformal confidence level: 90%
Additional frontier confidence levels: 80%, 85%, 90%, 92%, 95%
```

The run summary and reproducibility manifest are saved in:

```text
outputs_dp_shift_atcp_clinical_novelty/reports/run_summary.txt
outputs_dp_shift_atcp_clinical_novelty/reports/reproducibility_manifest.json
```

## Legacy Outputs

The older five-model adaptive transport calibration experiment remains available in:

```text
outputs_full_5models/
```

The current paper should cite the SHIFT-ATCP outputs in `outputs_dp_shift_atcp_clinical_novelty/`.

## Code Availability

Repository:

```text
https://github.com/diwakardkk/Confirmal_Prediction
```
