#!/usr/bin/env bash
# Run once from the project directory on the HPC login node.
set -euo pipefail

# If your centre uses environment modules, uncomment and adapt the next line.
# module load python/3.11

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import xgboost; print('XGBoost:', xgboost.__version__)"
