# SHIFT-ATCP Diabetes Conformal Prediction

This repository contains the source code for the SHIFT-ATCP diabetes conformal prediction experiments.

The project evaluates whether diabetes prediction models trained on a source cohort remain reliable when transported to external target populations. The main experiment uses Kaggle diabetes data as the source cohort, with NHANES and Pima Indians Diabetes data as external target cohorts.

## What Is Kept In Git

The GitHub repository is intentionally code-focused:

- `run_publication_ready_review_response.py`: current publication-ready experiment runner
- `run_full_adaptive_transport_5models.py`: legacy five-model experiment runner
- `requirements.txt`: Python dependencies
- `README.md`: project overview and run instructions

Generated outputs are not tracked in Git because they can become large and are reproducible from the scripts.

## Ignored Generated Files

The following local artifacts are ignored:

- `outputs*/`
- `overleaf*/`
- LaTeX files such as `*.tex` and `*.bib`
- model binaries such as `*.joblib`
- archives such as `*.zip`
- local report exports such as `*.docx` and `*.html`

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
