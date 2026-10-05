from __future__ import annotations

import itertools
import os

from common import FINAL_CONFIGS, RIDGE_SEARCH, ROOT, load_dataset, make_pipeline, polynomial_term_count

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict, cross_validate

RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"


def feature_sets(features: tuple[str, ...]) -> list[tuple[str, ...]]:
    return [
        subset
        for size in range(1, len(features) + 1)
        for subset in itertools.combinations(features, size)
    ]


def evaluate(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    degree: int,
    cv: KFold | RepeatedKFold,
    alpha: float = 0.0,
) -> dict[str, float | int | str]:
    scores = cross_validate(
        make_pipeline(degree, alpha),
        frame.loc[:, list(features)],
        frame["y"],
        cv=cv,
        scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
        n_jobs=-1,
    )
    fold_mse = -scores["test_mse"]
    return {
        "features": ",".join(features),
        "feature_count": len(features),
        "degree": degree,
        "alpha": alpha,
        "estimator": "OLS" if alpha == 0 else "Ridge",
        "polynomial_terms": polynomial_term_count(len(features), degree),
        "cv_mse_mean": float(fold_mse.mean()),
        "cv_mse_std": float(fold_mse.std(ddof=1)),
        "cv_r2_mean": float(scores["test_r2"].mean()),
        "cv_r2_std": float(scores["test_r2"].std(ddof=1)),
    }


def search_problem(
    problem: str,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    max_degree: int,
    cv: KFold,
    max_terms: int = 700,
) -> pd.DataFrame:
    rows = []
    for subset in feature_sets(features):
        for degree in range(1, max_degree + 1):
            if polynomial_term_count(len(subset), degree) > max_terms:
                break
            row = evaluate(frame, subset, degree, cv)
            row["problem"] = problem
            rows.append(row)
    return pd.DataFrame(rows)


def search_ridge(problem: str, frame: pd.DataFrame, cv: KFold) -> pd.DataFrame:
    features = FINAL_CONFIGS[problem]["features"]
    grid = RIDGE_SEARCH[problem]
    rows = []
    for degree in grid["degrees"]:
        for alpha in grid["alphas"]:
            row = evaluate(frame, features, degree, cv, alpha)
            row["problem"] = problem
            rows.append(row)
    return pd.DataFrame(rows)


def selected_metrics(
    problem: str,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    degree: int,
    alpha: float,
) -> tuple[dict[str, float | int | str], np.ndarray]:
    repeated_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    row = evaluate(frame, features, degree, repeated_cv, alpha)
    row["problem"] = problem

    diagnostic_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = cross_val_predict(
        make_pipeline(degree, alpha),
        frame.loc[:, list(features)],
        frame["y"],
        cv=diagnostic_cv,
        n_jobs=-1,
    )
    row["oof_mse"] = float(mean_squared_error(frame["y"], oof_predictions))
    row["oof_r2"] = float(r2_score(frame["y"], oof_predictions))

    fitted = make_pipeline(degree, alpha).fit(frame.loc[:, list(features)], frame["y"])
    train_predictions = fitted.predict(frame.loc[:, list(features)])
    row["training_mse"] = float(mean_squared_error(frame["y"], train_predictions))
    row["training_r2"] = float(r2_score(frame["y"], train_predictions))
    return row, oof_predictions


def plot_degree_search(ols_search: pd.DataFrame, ridge_search: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for axis, problem in zip(axes, ("var1", "var2")):
        config = FINAL_CONFIGS[problem]
        feature_label = ",".join(config["features"])
        values = ols_search.loc[
            (ols_search["problem"] == problem) & (ols_search["features"] == feature_label)
        ].sort_values("degree")
        ridge_values = ridge_search.loc[ridge_search["problem"] == problem]
        ridge_values = ridge_values.sort_values("cv_mse_mean").drop_duplicates("degree").sort_values("degree")
        axis.plot(values["degree"], values["cv_mse_mean"], marker="o", color="#1f5a85", label="OLS")
        axis.plot(ridge_values["degree"], ridge_values["cv_mse_mean"], marker="s", color="#317873", label="Best Ridge alpha")
        axis.axvline(config["degree"], color="#b23a48", linestyle="--", linewidth=1.4)
        axis.set_yscale("log")
        axis.set_title(f"{problem}: all assigned features")
        axis.set_xlabel("Polynomial degree")
        axis.set_ylabel("5-fold CV MSE (log scale)")
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "degree_selection.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_oof_diagnostics(
    frames: dict[str, pd.DataFrame], predictions: dict[str, np.ndarray]
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    for row_index, problem in enumerate(("var1", "var2")):
        actual = frames[problem]["y"].to_numpy()
        predicted = predictions[problem]
        residuals = actual - predicted

        scatter_axis = axes[row_index, 0]
        scatter_axis.scatter(actual, predicted, s=12, alpha=0.55, color="#1f5a85")
        limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
        scatter_axis.plot(limits, limits, color="#b23a48", linewidth=1.4)
        scatter_axis.set_title(f"{problem}: out-of-fold predictions")
        scatter_axis.set_xlabel("Actual y")
        scatter_axis.set_ylabel("Predicted y")
        scatter_axis.grid(alpha=0.2)

        residual_axis = axes[row_index, 1]
        residual_axis.scatter(predicted, residuals, s=12, alpha=0.55, color="#317873")
        residual_axis.axhline(0, color="#b23a48", linewidth=1.2)
        residual_axis.set_title(f"{problem}: residuals")
        residual_axis.set_xlabel("Predicted y")
        residual_axis.set_ylabel("Actual - predicted")
        residual_axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "oof_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    frames = {problem: load_dataset(problem, "train") for problem in ("var1", "var2")}
    screening_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    var1_search = search_problem(
        "var1",
        frames["var1"],
        tuple(f"x{i}" for i in range(1, 7)),
        10,
        screening_cv,
    )
    var2_search = search_problem(
        "var2",
        frames["var2"],
        ("x1", "x2", "x3"),
        20,
        screening_cv,
    )
    ols_search = pd.concat([var1_search, var2_search], ignore_index=True)
    ols_search = ols_search.sort_values(
        ["problem", "cv_mse_mean", "feature_count", "degree"], ignore_index=True
    )
    ols_search.to_csv(RESULTS_DIR / "model_selection.csv", index=False)

    ridge_search = pd.concat(
        [search_ridge(problem, frames[problem], screening_cv) for problem in ("var1", "var2")],
        ignore_index=True,
    ).sort_values(["problem", "cv_mse_mean", "degree", "alpha"], ignore_index=True)
    ridge_search.to_csv(RESULTS_DIR / "ridge_selection.csv", index=False)

    metric_rows = []
    oof_predictions = {}
    for problem, config in FINAL_CONFIGS.items():
        row, predictions = selected_metrics(
            problem, frames[problem], config["features"], config["degree"], config["alpha"]
        )
        metric_rows.append(row)
        oof_predictions[problem] = predictions
    metrics = pd.DataFrame(metric_rows).sort_values("problem")
    metrics.to_csv(RESULTS_DIR / "final_metrics.csv", index=False)

    plot_degree_search(ols_search, ridge_search)
    plot_oof_diagnostics(frames, oof_predictions)

    print("Best OLS screening configurations:")
    print(ols_search.groupby("problem", sort=False).head(3).to_string(index=False))
    print("\nBest Ridge screening configurations:")
    print(ridge_search.groupby("problem", sort=False).head(5).to_string(index=False))
    print("\nRepeated cross-validation metrics for selected models:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
