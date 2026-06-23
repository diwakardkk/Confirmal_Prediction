"""Split-safe model preprocessing factories."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense, version-compatible one-hot encoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical predictors from a feature frame."""
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    """Build a fit-on-training-only imputation, encoding, and optional scaling pipeline."""
    numeric, categorical = infer_feature_types(X)
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers: list[tuple[str, object, Iterable[str]]] = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]),
            categorical,
        ))
    if not transformers:
        raise ValueError("No usable predictor columns are available.")
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return robust output names after a fitted ColumnTransformer."""
    try:
        return [str(value) for value in preprocessor.get_feature_names_out()]
    except Exception:
        names: list[str] = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder" or transformer == "drop":
                continue
            if name == "numeric":
                names.extend(map(str, columns))
            elif name == "categorical":
                encoder = transformer.named_steps["onehot"]
                names.extend(map(str, encoder.get_feature_names_out(columns)))
        return names
