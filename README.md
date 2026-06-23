# Reliable Diabetes Risk Prediction Using Calibrated XGBoost and Conformal Prediction

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
