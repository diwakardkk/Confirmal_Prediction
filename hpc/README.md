# HPC execution (Slurm)

These scripts target a Slurm-managed CPU cluster. They run the primary analysis (including the secondary balanced sensitivity analysis) and the BRFSS 50/50 analysis as separate jobs, so their outputs and logs cannot overwrite one another.

## Expected layout on the cluster

Keep the project beside the datasets:

```text
Final_mycode_CP/
├── diabetes_prediction_dataset.csv
├── _Datasets_/
│   └── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
└── diabetes_conformal_revised/
    ├── hpc/
    └── run_pipeline.py
```

If your files live elsewhere, export the directory containing `diabetes_prediction_dataset.csv` before submitting:

```bash
export DATA_ROOT=/absolute/path/to/Final_mycode_CP
```

## First-time setup

Copy the complete project and datasets to the cluster. On a login node:

```bash
cd /path/to/Final_mycode_CP/diabetes_conformal_revised
bash hpc/setup_environment.sh
```

If the cluster uses modules, edit the commented `module load` line in `setup_environment.sh` before running it. Use a supported Python version (3.10–3.12 is generally safest for the scientific stack).

## Submit jobs

From the project directory:

```bash
sbatch hpc/run_primary_with_sensitivity.sbatch
sbatch hpc/run_brfss_5050.sbatch
```

Monitor jobs:

```bash
squeue -u "$USER"
```

Logs are written to `logs/`; outputs are written separately to:

- `outputs_hpc_primary_with_balanced_sensitivity/`
- `outputs_hpc_brfss_5050_balanced/`

## Resource settings

The scripts request one node, 16 CPUs, 64 GB RAM, and 24 hours. Adjust `--partition`, `--account`, CPU, RAM, and wall time to your cluster policy. The pipeline is CPU-based; do not request a GPU unless the code and cluster have a verified CUDA-enabled XGBoost environment.
