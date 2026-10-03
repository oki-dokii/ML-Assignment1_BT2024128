# Polynomial Regression Assignment - BT2024128

This repository contains the complete solution for both personalized polynomial-regression problems in ML Assignment 1.

## Final models

| Problem | Features | Degree | Polynomial terms | Repeated 5-fold CV MSE | Repeated 5-fold CV R2 |
|---|---|---:|---:|---:|---:|
| var1 | x1, x2, x3, x4, x5, x6 | 4 | 209 | 0.7814 +/- 0.1174 | 0.9273 +/- 0.0134 |
| var2 | x1, x2, x3 | 8 | 164 | 0.2574 +/- 0.0309 | 0.9938 +/- 0.0014 |

The models use `PolynomialFeatures`, followed by `StandardScaler` and ordinary least-squares `LinearRegression`. All preprocessing is inside a scikit-learn pipeline, so it is fitted independently within every cross-validation fold.

## Repository structure

```text
data/BT2024128/       Training and test datasets
src/common.py         Shared paths, validation and model construction
src/model_selection.py Feature/degree search, metrics and diagnostic plots
src/train_predict.py  Final training, model serialization and test prediction
results/              Cross-validation results
plots/                Degree-selection and out-of-fold diagnostic figures
models/               Fitted scikit-learn pipelines
predictions/          Submission-ready CSV files
report/               Assignment report source and PDF
```

## Setup

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Reproduce model selection

Run from the repository root:

```bash
python3 src/model_selection.py
```

This performs exhaustive feature-subset screening with five-fold cross-validation. It searches up to the assignment limits of degree 10 for var1 and degree 20 for var2, while discarding configurations with more than 700 expanded polynomial terms. The selected models are then assessed with repeated five-fold cross-validation using five repeats. The script writes:

- `results/model_selection.csv`
- `results/final_metrics.csv`
- `plots/degree_selection.png`
- `plots/oof_diagnostics.png`

Configurations were excluded once their polynomial expansion exceeded 700 terms. A five-fold training split contains 800 observations, so larger expansions approach or exceed the number of available fitting observations and become poorly determined.

## Generate predictions

```bash
python3 src/train_predict.py
```

The script refits each selected pipeline on all 1,000 training observations, saves the fitted pipelines, and creates:

- `predictions/BT2024128_pred_var1.csv`
- `predictions/BT2024128_pred_var2.csv`

Each prediction file contains exactly one column named `y`, has 1,000 rows, preserves test-set order, and does not contain an index column.

## Validation design

- Model selection uses training data only.
- Test data is never used to tune the features or polynomial degree.
- MSE is the primary selection metric and R2 is reported as a complementary metric.
- Scaling occurs after polynomial expansion and is learned inside each fold, preventing preprocessing leakage.
- Fixed random seeds make the reported splits reproducible.
