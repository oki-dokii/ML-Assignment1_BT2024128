from __future__ import annotations

from math import comb
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROLL_NUMBER = "BT2024128"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / ROLL_NUMBER

FINAL_CONFIGS = {
    "var1": {"features": ("x1", "x2", "x3", "x4", "x5", "x6"), "degree": 4},
    "var2": {"features": ("x1", "x2", "x3"), "degree": 8},
}

EXPECTED_COLUMNS = {
    "var1": {
        "train": ("x1", "x2", "x3", "x4", "x5", "x6", "y"),
        "test": ("x1", "x2", "x3", "x4", "x5", "x6"),
    },
    "var2": {
        "train": ("x1", "x2", "x3", "y"),
        "test": ("x1", "x2", "x3"),
    },
}


def dataset_path(problem: str, split: str) -> Path:
    return DATA_DIR / f"{ROLL_NUMBER}_{split}_{problem}.csv"


def load_dataset(problem: str, split: str) -> pd.DataFrame:
    path = dataset_path(problem, split)
    frame = pd.read_csv(path)
    expected = list(EXPECTED_COLUMNS[problem][split])
    if list(frame.columns) != expected:
        raise ValueError(
            f"Unexpected columns in {path.name}: {list(frame.columns)}; expected {expected}"
        )
    if frame.empty:
        raise ValueError(f"{path.name} is empty")
    if frame.isna().any().any():
        raise ValueError(f"{path.name} contains missing values")
    if not all(pd.api.types.is_numeric_dtype(frame[column]) for column in frame.columns):
        raise TypeError(f"{path.name} contains non-numeric columns")
    return frame


def make_pipeline(degree: int) -> Pipeline:
    return Pipeline(
        [
            ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("regression", LinearRegression()),
        ]
    )


def polynomial_term_count(feature_count: int, degree: int) -> int:
    return comb(feature_count + degree, degree) - 1
