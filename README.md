# Diabetes Status Classification with Calibrated XGBoost and Conformal Prediction

This project regenerates reproducible tables, figures, fitted models, logs, and a manifest for an IEEE Access manuscript. The primary analysis uses the cleaned dataset at its observed class prevalence. Optional balanced under-sampling is strictly a secondary sensitivity analysis.

## Installation

```bash
cd diabetes_conformal_revised
python3 -m pip install -r requirements.txt
```

`shap` is optional at execution time. If it is unavailable or incompatible, the pipeline writes a clear skip report and continues with permutation importance.

## Full experiment

When the dataset is in `Diabetes/diabetes_prediction_dataset.csv` relative to this folder:

```bash
python3 run_pipeline.py --data Diabetes/diabetes_prediction_dataset.csv --outdir outputs --target diabetes --seed 42 --device auto --dpi 600 --run-balanced-sensitivity
```

For the dataset currently located one directory above this project folder:

```bash
python3 run_pipeline.py --data ../diabetes_prediction_dataset.csv --outdir outputs --target diabetes --seed 42 --device auto --dpi 600 --run-balanced-sensitivity
```

The full search is intentionally exhaustive and can take substantial time. Test your environment first:

```bash
python3 run_pipeline.py --data ../diabetes_prediction_dataset.csv --outdir outputs_quick --target diabetes --seed 42 --device cpu --dpi 600 --quick
```

## BRFSS 2015 balanced dataset

To analyze the externally supplied 50/50 BRFSS dataset without mixing its artefacts with the primary dataset, use a separate output directory and label. Do **not** add `--run-balanced-sensitivity`: the selected input file is already the balanced BRFSS dataset.

```bash
python3 run_pipeline.py \
  --data ../_Datasets_/diabetes_binary_5050split_health_indicators_BRFSS2015.csv \
  --outdir outputs_brfss_5050_balanced \
  --target Diabetes_binary \
  --seed 42 \
  --device cpu \
  --dpi 600 \
  --experiment-label brfss_5050_balanced \
  --analysis-description "BRFSS 2015 externally supplied balanced 50/50 dataset"
```

## Pima Indians dataset

The standard Pima file uses `Outcome` as its binary target. In the Pima profile, zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are treated as unavailable physiological measurements and converted to missing values before split-safe median imputation. Zero pregnancies remain valid and are retained.

```bash
python3 run_pipeline.py \
  --data ../diabetes.csv \
  --outdir outputs_pima_secondary \
  --target Outcome \
  --dataset-profile pima \
  --seed 42 \
  --device cpu \
  --dpi 600 \
  --bootstrap-repeats 1000 \
  --experiment-label pima_secondary \
  --analysis-description "Pima Indians Diabetes Database secondary benchmark analysis"
```

The Pima analysis is a separate benchmark study. Its population, predictors, outcome ascertainment, and observed prevalence differ from the primary dataset; therefore, its performance metrics should not be pooled with or described as direct external validation of the original model.

## NHANES 2017--2018 external transportability evaluation

The dedicated NHANES workflow creates a reduced model using only the five features harmonized across the source and NHANES data: sex, age, BMI, HbA1c, and fasting glucose. The reduced model is trained and calibrated only on the source dataset, then applied unchanged to the NHANES cohort. NHANES outcomes are derived from `DIQ010` self-reported clinician-diagnosed diabetes; this is an external transportability analysis, not direct validation of the full primary model.

Required NHANES files in one directory: `DEMO_J.xpt`, `DIQ_J.xpt`, `BMX_J.xpt`, `BPX_J.xpt`, `GHB_J.xpt`, and `GLU_J.xpt`.

```bash
python3 run_nhanes_external_validation.py \
  --source-data ../diabetes_prediction_dataset.csv \
  --nhanes-dir ../NHANES \
  --outdir outputs_nhanes_external \
  --seed 42 \
  --device cpu \
  --dpi 600 \
  --bootstrap-repeats 1000
```

If `--data` is omitted, the runner searches likely CSV names in the current folder, this project folder, and its parent folder.

## Output structure

- `outputs/data/`: cleaned datasets and one fixed train/calibration/test partition.
- `outputs/tables/`: manuscript tables in CSV and XLSX, plus detailed CV and curve data.
- `outputs/figures/`: 600-dpi PNG figures.
- `outputs/models/`: tuned sklearn pipelines saved with joblib.
- `outputs/reports/`: classification reports, hyperparameters, SHAP status, and manuscript hand-off summary.
- `outputs/logs/`: `pipeline.log`.
- `outputs/reproducibility_manifest.json`: dataset fingerprint, versions, seed, features, best parameters, device, and generated artefacts.

The primary manuscript results use files suffixed `original_prevalence`. Files suffixed `balanced_sensitivity` are secondary only and must not replace the primary results.

## Interpretation and reporting boundaries

This repository implements **diabetes-status classification**, not a prospective diagnostic replacement or autonomous clinical decision system. HbA1c and blood glucose are closely related to diabetes assessment and are intentionally retained as available predictors; results must therefore not be described as pre-diagnostic risk prediction.

The primary model is evaluated internally on the original-prevalence source dataset. Pima is a secondary benchmark only. NHANES is an external transportability evaluation of a separately trained, five-feature harmonized reduced model; it is not validation of the full primary eight-feature model.

The completed NHANES experiment demonstrated strong external discrimination but reduced calibration quality and incomplete transport of source-derived operating thresholds and conformal coverage. In particular, a source-selected threshold targeting 95% sensitivity achieved 72.46% sensitivity in NHANES, and source-calibrated Mondrian conformal prediction achieved 70.22% diabetes-class coverage at nominal 90% confidence. These results should be reported as evidence that target-domain recalibration is required before use in a new population.

## Reliability and clinical robustness analyses

Each complete run additionally produces:

- Standard split-conformal and class-conditional (Mondrian) conformal prediction tables.
- Five-fold out-of-fold training-only selection of clinical operating thresholds targeting 85%, 90%, and 95% sensitivity, evaluated only once on the untouched test set.
- Stratified percentile bootstrap 95% confidence intervals for point-prediction and conformal metrics.
- Descriptive test-set subgroup reporting by available sex/gender, age, hypertension, and heart-disease variables.

The default is 1,000 bootstrap replicates. Use `--bootstrap-repeats 1000` explicitly in a manuscript run; retain the same number in the paper and reproducibility manifest.

## HPC execution

Slurm submission scripts are provided in [`hpc/`](hpc/README.md). They run the original dataset and BRFSS dataset as independent CPU jobs with separate output folders and logs.

## Conformal prediction

The selected XGBoost configuration is probability-calibrated with five-fold cross-validation within the training split. The held-out calibration split is then used exclusively for split-conformal nonconformity scores, `1 - p(true class | x)`. For a test record, both class probabilities are considered when constructing its prediction set. Empirical coverage is the proportion of test records whose true label is contained in that set.

Use only generated tables and figures for manuscript revision; do not reuse older hardcoded manuscript values.
