"""File, logging, and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd


OUTPUT_SUBDIRECTORIES = ("data", "tables", "figures", "models", "reports", "logs")


def ensure_output_directories(outdir: Path) -> dict[str, Path]:
    """Create and return the standard output directory layout."""
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {"root": outdir}
    for name in OUTPUT_SUBDIRECTORIES:
        path = outdir / name
        path.mkdir(exist_ok=True)
        paths[name] = path
    return paths


def configure_logging(log_path: Path) -> logging.Logger:
    """Configure one console/file logger without duplicate handlers."""
    logger = logging.getLogger("diabetes_conformal")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def safe_stem(value: str) -> str:
    """Return a portable lower-case filename stem."""
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def write_table(table: pd.DataFrame, csv_path: Path, logger: logging.Logger) -> None:
    """Always write CSV and attempt a matching XLSX without aborting the run."""
    table.to_csv(csv_path, index=False)
    try:
        table.to_excel(csv_path.with_suffix(".xlsx"), index=False)
    except Exception as exc:  # openpyxl is optional at execution time.
        logger.warning("Could not write Excel table %s: %s", csv_path.with_suffix(".xlsx").name, exc)


def write_json(payload: Any, path: Path) -> None:
    """Write JSON with support for common NumPy/Pandas scalar types."""
    def default(value: Any) -> Any:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, (Path, datetime)):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, default=default) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    """Calculate a streaming SHA-256 checksum."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_model(model: Any, path: Path) -> None:
    """Persist a fitted model using joblib."""
    joblib.dump(model, path)


def package_versions() -> dict[str, str | None]:
    """Return versions of packages relevant to reproducibility."""
    import matplotlib
    import sklearn
    import xgboost

    versions: dict[str, str | None] = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "matplotlib": matplotlib.__version__,
        "joblib": joblib.__version__,
    }
    try:
        import seaborn
        versions["seaborn"] = seaborn.__version__
    except ImportError:
        versions["seaborn"] = None
    try:
        import shap
        versions["shap"] = shap.__version__
    except ImportError:
        versions["shap"] = None
    return versions


def base_manifest() -> dict[str, Any]:
    """Return platform metadata common to every reproducibility manifest."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
    }


def list_relative_files(root: Path) -> list[str]:
    """List generated artefacts relative to an output root."""
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())
