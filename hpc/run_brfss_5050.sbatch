#!/usr/bin/env bash
# Submit with: sbatch hpc/run_brfss_5050.sbatch
#SBATCH --job-name=diabetes-brfss
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nodes=1

# Add the required partition/account directives for your cluster, for example:
# Example directive: #SBATCH --partition=compute
# Example directive: #SBATCH --account=YOUR_ALLOCATION

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
DATA_ROOT="${DATA_ROOT:-$(cd "$PROJECT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

source .venv/bin/activate
mkdir -p logs .cache/matplotlib
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export PYTHONHASHSEED=42
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

python run_pipeline.py \
  --data "$DATA_ROOT/_Datasets_/diabetes_binary_5050split_health_indicators_BRFSS2015.csv" \
  --outdir outputs_hpc_brfss_5050_balanced \
  --target Diabetes_binary \
  --seed 42 \
  --device cpu \
  --dpi 600 \
  --experiment-label brfss_5050_balanced \
  --analysis-description "BRFSS 2015 externally supplied balanced 50/50 dataset"
