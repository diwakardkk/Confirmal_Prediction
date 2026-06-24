"""NHANES 2017--2018 assembly and transparent feature harmonization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_FILES = ("DEMO_J.xpt", "DIQ_J.xpt", "BMX_J.xpt", "BPX_J.xpt", "GHB_J.xpt", "GLU_J.xpt")
HARMONIZED_FEATURES = ("sex", "age", "bmi", "hba1c", "glucose")


def _find_file(directory: Path, filename: str) -> Path:
    """Find an NHANES XPT file without relying on its filename capitalization."""
    matches = [path for path in directory.iterdir() if path.name.casefold() == filename.casefold()]
    if not matches:
        raise FileNotFoundError(f"Required NHANES file {filename} is absent from {directory}.")
    return matches[0]


def _read_xpt(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read selected XPT variables and normalize the participant ID."""
    frame = pd.read_sas(path, format="xport")
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    frame = frame[columns].copy()
    frame["SEQN"] = frame["SEQN"].astype("Int64")
    return frame


def build_nhanes_2017_2018(nhanes_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Merge public NHANES files into an adult, harmonized external cohort.

    Outcome definition: DIQ010=1 (self-reported clinician-diagnosed diabetes)
    versus DIQ010=2 (no reported diagnosis). Borderline, unknown, refusal, and
    missing responses are excluded. Known pregnancy at examination is excluded.
    The fasting-glucose requirement restricts the cohort to the fasting subsample.
    """
    nhanes_dir = Path(nhanes_dir).expanduser().resolve()
    if not nhanes_dir.is_dir():
        raise NotADirectoryError(f"NHANES directory not found: {nhanes_dir}")
    files = {name: _find_file(nhanes_dir, name) for name in REQUIRED_FILES}
    demo = _read_xpt(files["DEMO_J.xpt"], ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDEXPRG", "WTMEC2YR"])
    diq = _read_xpt(files["DIQ_J.xpt"], ["SEQN", "DIQ010"])
    bmx = _read_xpt(files["BMX_J.xpt"], ["SEQN", "BMXBMI"])
    bpx = _read_xpt(files["BPX_J.xpt"], ["SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1", "BPXDI2", "BPXDI3"])
    ghb = _read_xpt(files["GHB_J.xpt"], ["SEQN", "LBXGH"])
    glu = _read_xpt(files["GLU_J.xpt"], ["SEQN", "WTSAF2YR", "LBXGLU"])

    merged = demo.merge(diq, on="SEQN", how="inner").merge(bmx, on="SEQN", how="inner")
    merged = merged.merge(bpx, on="SEQN", how="left").merge(ghb, on="SEQN", how="inner")
    merged = merged.merge(glu, on="SEQN", how="inner")
    n_merged = len(merged)
    adult = merged[merged["RIDAGEYR"] >= 20].copy()
    n_adult = len(adult)
    pregnancy_excluded = int(adult["RIDEXPRG"].eq(1).sum())
    adult = adult.loc[~adult["RIDEXPRG"].eq(1)].copy()
    outcome_valid = adult["DIQ010"].isin([1, 2])
    n_outcome_excluded = int((~outcome_valid).sum())
    adult = adult.loc[outcome_valid].copy()

    sex_map = {1.0: "Male", 2.0: "Female"}
    cohort = pd.DataFrame({
        "SEQN": adult["SEQN"].astype(int),
        "sex": adult["RIAGENDR"].map(sex_map),
        "age": adult["RIDAGEYR"],
        "bmi": adult["BMXBMI"],
        "hba1c": adult["LBXGH"],
        "glucose": adult["LBXGLU"],
        "diabetes": adult["DIQ010"].map({1.0: 1, 2.0: 0}),
        "fasting_subsample_weight": adult["WTSAF2YR"],
        "mec_exam_weight": adult["WTMEC2YR"],
    })
    complete_mask = cohort[list(HARMONIZED_FEATURES) + ["diabetes", "fasting_subsample_weight"]].notna().all(axis=1)
    n_incomplete = int((~complete_mask).sum())
    cohort = cohort.loc[complete_mask].reset_index(drop=True)
    cohort["diabetes"] = cohort["diabetes"].astype(int)
    mapping = pd.DataFrame([
        {"harmonized_feature": "sex", "source_dataset_variable": "gender", "nhanes_variable": "RIAGENDR", "notes": "Female/Male labels harmonized."},
        {"harmonized_feature": "age", "source_dataset_variable": "age", "nhanes_variable": "RIDAGEYR", "notes": "Age in years."},
        {"harmonized_feature": "bmi", "source_dataset_variable": "bmi", "nhanes_variable": "BMXBMI", "notes": "Body mass index in kg/m^2."},
        {"harmonized_feature": "hba1c", "source_dataset_variable": "HbA1c_level", "nhanes_variable": "LBXGH", "notes": "Glycohemoglobin percentage."},
        {"harmonized_feature": "glucose", "source_dataset_variable": "blood_glucose_level", "nhanes_variable": "LBXGLU", "notes": "Fasting plasma glucose in mg/dL; fasting subsample."},
        {"harmonized_feature": "diabetes", "source_dataset_variable": "diabetes", "nhanes_variable": "DIQ010", "notes": "NHANES self-reported clinician-diagnosed diabetes: 1=yes, 2=no; other responses excluded."},
    ])
    audit = {
        "cycle": "NHANES 2017-2018", "files_merged": len(files), "rows_after_required_file_merge": n_merged,
        "adults_age_20_or_older": n_adult, "known_pregnancy_excluded": pregnancy_excluded,
        "invalid_or_borderline_outcome_excluded": n_outcome_excluded,
        "incomplete_harmonized_rows_excluded": n_incomplete, "final_external_cohort": len(cohort),
        "class_0_count": int((cohort["diabetes"] == 0).sum()), "class_1_count": int((cohort["diabetes"] == 1).sum()),
        "diabetes_prevalence_percent": float(cohort["diabetes"].mean() * 100),
        "outcome_definition": "DIQ010=1 versus DIQ010=2; borderline/unknown/refused/missing responses excluded.",
        "analysis_type": "External transportability evaluation of a harmonized reduced model; not validation of the full primary model.",
    }
    return cohort, mapping, audit


def build_source_harmonized(source: pd.DataFrame) -> pd.DataFrame:
    """Map the primary diabetes dataset to the common five-feature schema."""
    required = {"gender", "age", "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Primary data cannot be harmonized; missing columns: {sorted(missing)}")
    frame = pd.DataFrame({
        "sex": source["gender"].replace({"Female": "Female", "Male": "Male"}),
        "age": source["age"], "bmi": source["bmi"], "hba1c": source["HbA1c_level"],
        "glucose": source["blood_glucose_level"], "diabetes": source["diabetes"].astype(int),
    })
    if frame["sex"].isna().any():
        raise ValueError("Primary harmonized data contain unmapped sex values.")
    return frame
