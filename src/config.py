"""Shared configuration for the diabetes analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    """Configuration passed through one analysis run."""

    data_path: Path
    outdir: Path
    target: str = "diabetes"
    dataset_profile: str = "generic"
    seed: int = 42
    device_requested: str = "auto"
    dpi: int = 600
    bootstrap_repeats: int = 1000
    quick: bool = False
    run_balanced_sensitivity: bool = False


CONFIDENCE_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
SENSITIVITY_CONFIDENCES = (0.80, 0.90, 0.95)
SENSITIVITY_FRACTIONS = (0.10, 0.20, 0.30, 0.50, 1.00)
CLINICAL_TARGET_SENSITIVITIES = (0.85, 0.90, 0.95)
BOOTSTRAP_CONFIDENCE = 0.95
MODEL_ORDER = (
    "Logistic Regression",
    "SVM",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
)
