# Adaptive Transport Calibration for Diabetes Prediction

This repository contains the current 5-model adaptive transport calibration experiment for diabetes risk prediction. The source model is trained on the Kaggle diabetes dataset and externally evaluated on NHANES and Pima. The analysis measures dataset shift, probability calibration, external transportability, adaptive transport conformal prediction, subgroup reliability, and Transport Reliability Score rankings.

## Models

The corrected full experiment evaluates five model families across the same three feature sets:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

## Run

Install the dependencies, including XGBoost, LightGBM, and CatBoost:

```bash
python3 -m pip install -r requirements.txt
```

Run the full current experiment:

```bash
python3 run_full_adaptive_transport_5models.py
```

All current outputs are written to:

```text
outputs_full_5models/
```

## Main Outputs

- `outputs_full_5models/data/`: harmonized Kaggle, NHANES, and Pima datasets used by the experiment.
- `outputs_full_5models/tables/`: dataset shift, internal model metrics, calibration rankings, external transportability, conformal coverage, fairness, bootstrap confidence intervals, McNemar tests, and TRS tables.
- `outputs_full_5models/figures/`: publication-style 500 dpi figures with enlarged fonts.
- `outputs_full_5models/models/`: fitted source models.
- `outputs_full_5models/reports/`: run summary and reproducibility manifest.
- `overleaf_fresh_current_5model_paper/`: LaTeX paper source and figures for the current results.

The current figures include corrected feature histogram panels, clearer conformal coverage visualization, and compact TRS ranking plots for NHANES and Pima.
