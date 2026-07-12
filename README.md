# SHIFT-ATCP Diabetes Conformal Prediction

This repository contains the source code for the paper:

**SHIFT-ATCP: Shift-Aware and Diabetes-Protected Adaptive Conformal Prediction for Reliable Diabetes Classification Under Dataset Shift**

The project evaluates whether diabetes prediction models trained on a source cohort remain reliable when transported to external target populations. The main experiment uses Kaggle diabetes data as the source cohort, with NHANES 2017-2018 and Pima Indians Diabetes data as external target cohorts.

## Dataset Details

Three public diabetes-related datasets are used after cleaning and harmonization:

| Dataset | Role | N | Class 0 | Class 1 | Diabetes prevalence |
|---|---:|---:|---:|---:|---:|
| Kaggle Diabetes Prediction Dataset | Source development cohort | 96,128 | 87,646 | 8,482 | 8.82% |
| NHANES 2017-2018 | Main external target cohort | 2,291 | 1,888 | 403 | 17.59% |
| Pima Indians Diabetes Dataset | Extreme-shift target cohort | 768 | 500 | 268 | 34.90% |

The final target evaluation uses held-out target-test partitions. Target calibration data are used for calibration and conformal operating-point selection, while target-test data are reserved for final reporting.

## Clinical Referral And Cost Figure

![Clinical referral and cost profile of SHIFT-ATCP variants](docs/figures/figure_shift_atcp_clinical_referral_cost.png)

This figure summarizes the clinical referral and cost profile of the main SHIFT-ATCP variants. Diabetes-protected methods increase uncertainty and referral when needed, while improving diabetes-positive coverage under dataset shift.

## What Is Kept In Git

The GitHub repository is intentionally code-focused:

- `run_publication_ready_review_response.py`: current publication-ready experiment runner
- `run_full_adaptive_transport_5models.py`: legacy five-model experiment runner
- `requirements.txt`: Python dependencies
- `README.md`: project overview and run instructions

Generated outputs are not tracked in Git because they can become large and are reproducible from the scripts.




## Run The Current Experiment

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the current experiment:

```bash
python3 run_publication_ready_review_response.py
```

The script writes generated outputs locally under:

```text
outputs_dp_shift_atcp_clinical_novelty/
```

Those files are useful for local analysis, but they should not be committed to GitHub.

## Main Method Summary

SHIFT-ATCP estimates source trust under dataset shift using distribution-shift and calibration signals. DP-SHIFT-ATCP adds diabetes-positive protection with class-specific conformal thresholds so minority-class coverage is evaluated explicitly, not hidden by marginal coverage.

Repository:

```text
https://github.com/diwakardkk/Confirmal_Prediction
```
