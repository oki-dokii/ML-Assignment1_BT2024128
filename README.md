# Polynomial Regression Assignment - BT2024128

This repository contains the complete solution for both personalized polynomial-regression problems in ML Assignment 1.

## Final models

| Problem | Features | Degree | Ridge alpha | Polynomial terms | Repeated 5-fold CV MSE | Repeated 5-fold CV R2 |
|---|---|---:|---:|---:|---:|---:|
| var1 | x1, x2, x3, x4, x5, x6 | 5 | 10 | 461 | 0.4378 +/- 0.0502 | 0.9593 +/- 0.0063 |
| var2 | x1, x2, x3 | 12 | 1 | 454 | 0.2291 +/- 0.0251 | 0.9945 +/- 0.0013 |

The models use `PolynomialFeatures`, followed by `StandardScaler` and `Ridge`. Ridge adds L2 regularization to the polynomial coefficients. It improved five-fold CV MSE over the best ordinary least-squares baselines: 0.7725 to 0.4265 for var1 and 0.2544 to 0.2268 for var2. All preprocessing is inside a scikit-learn pipeline, so it is fitted independently within every cross-validation fold.

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

This first performs exhaustive feature-subset screening with ordinary least squares and five-fold cross-validation. It searches up to the assignment limits of degree 10 for var1 and degree 20 for var2, discarding OLS configurations with more than 700 expanded polynomial terms. Both screens favored all available inputs. It then searches Ridge penalties and degrees on those full feature sets: degrees 3-6 for var1 and 7-16 for var2. The selected models are assessed with repeated five-fold cross-validation using five repeats. The script writes:

- `results/model_selection.csv` (OLS feature and degree screen)
- `results/ridge_selection.csv` (Ridge degree and penalty screen)
- `results/final_metrics.csv`
- `plots/degree_selection.png`
- `plots/oof_diagnostics.png`

The 700-term restriction applies only to the OLS screen. Ridge can fit larger expansions because its penalty stabilizes the coefficients. The reported validation scores estimate performance on unseen observations; the hidden test targets were unavailable.

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
- Ridge degree and alpha were chosen by the same five-fold search; the fixed configurations were checked again using 25 repeated validation folds.
- Scaling occurs after polynomial expansion and is learned inside each fold, preventing preprocessing leakage.
- Fixed random seeds make the reported splits reproducible.
