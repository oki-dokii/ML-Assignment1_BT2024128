# Polynomial Regression Assignment - BT2024128

This repository contains the complete solution for both personalized polynomial-regression problems in ML Assignment 1.

## Final models

| Problem | Final pipeline | Features | Degree | Lasso alpha | Ridge alpha | Terms retained | Repeated 5-fold CV MSE | Repeated 5-fold CV R2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| var1 | Polynomial, scale, Lasso selection, Ridge | x1, x2, x3, x4, x5, x6 | 5 | 0.03 | 0.1 | 69 of 461 | 0.2975 +/- 0.0296 | 0.9724 +/- 0.0036 |
| var2 | Polynomial, scale, Ridge | x1, x2, x3 | 12 | - | 1 | 454 | 0.2291 +/- 0.0251 | 0.9945 +/- 0.0013 |

For var1, Lasso selects polynomial terms before Ridge refits their coefficients. This reduced five-fold CV MSE from 0.4265 for the earlier Ridge-only model to 0.3082. An inner-three-fold/outer-five-fold nested search gave MSE 0.3086. Lasso selection did not improve var2 in the completed comparisons, so its final model remains Ridge-only. All preprocessing and feature selection are inside scikit-learn pipelines, so they are refitted independently within each validation fold.

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

This first performs exhaustive feature-subset screening with ordinary least squares and five-fold cross-validation. It searches up to the assignment limits of degree 10 for var1 and degree 20 for var2, discarding OLS configurations with more than 700 expanded polynomial terms. Both screens favored all available inputs. It then searches Ridge penalties and degrees on those full feature sets: degrees 3-6 for var1 and 7-16 for var2. For var1, it additionally searches Lasso term-selection strengths 0.003, 0.01 and 0.03, each followed by Ridge strengths 0.1, 1, 10 and 30 at degree 5. It checks this selection strategy with nested cross-validation. The final fixed models are also assessed with repeated five-fold cross-validation using five repeats. The script writes:

- `results/model_selection.csv` (OLS feature and degree screen)
- `results/ridge_selection.csv` (Ridge degree and penalty screen)
- `results/lasso_ridge_selection.csv` (var1 Lasso-then-Ridge screen)
- `results/lasso_nested_cv.csv` (var1 nested-validation folds)
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
- Degree and penalty strengths were chosen using five-fold validation; var1 Lasso/Ridge tuning was also checked with nested validation. The fixed configurations were checked again using 25 repeated validation folds.
- Scaling occurs after polynomial expansion; Lasso selection, when used, is also learned inside each fold to prevent leakage.
- Fixed random seeds make the reported splits reproducible.
