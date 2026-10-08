from __future__ import annotations

from math import comb
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROLL_NUMBER = "BT2024128"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / ROLL_NUMBER

FINAL_CONFIGS = {
    "var1": {"features": ("x1", "x2", "x3", "x4", "x5", "x6"), "degree": 5, "alpha": 0.1, "lasso_alpha": 0.03},
    "var2": {"features": ("x1", "x2", "x3"), "degree": 12, "alpha": 1.0, "lasso_alpha": 0.0},
}

RIDGE_SEARCH = {
    "var1": {"degrees": range(3, 7), "alphas": (0.1, 1.0, 3.0, 10.0, 30.0, 100.0)},
    "var2": {"degrees": range(7, 17), "alphas": (0.01, 0.1, 1.0, 3.0, 10.0)},
}

LASSO_RIDGE_SEARCH = {
    "lasso_alphas": (0.003, 0.01, 0.03),
    "ridge_alphas": (0.1, 1.0, 10.0, 30.0),
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


def make_pipeline(degree: int, alpha: float = 0.0, lasso_alpha: float = 0.0) -> Pipeline:
    if alpha < 0 or lasso_alpha < 0:
        raise ValueError("Regularization strengths must be nonnegative")
    regressor = (
        LinearRegression()
        if alpha == 0
        else Ridge(alpha=alpha, solver="lsqr", tol=1e-7, max_iter=20000)
    )
    steps = [
        ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
    ]
    if lasso_alpha > 0:
        steps.append(
            (
                "selector",
                SelectFromModel(
                    Lasso(alpha=lasso_alpha, max_iter=10000, tol=1e-4),
                    threshold=1e-8,
                ),
            )
        )
    steps.append(("regression", regressor))
    return Pipeline(steps)


def polynomial_term_count(feature_count: int, degree: int) -> int:
    return comb(feature_count + degree, degree) - 1
