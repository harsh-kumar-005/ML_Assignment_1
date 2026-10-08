# Polynomial Regression: Assignment 1 (BT2024008)

Two regression problems are solved with polynomial feature expansion followed by regularised linear regression. Model degree and regularisation are selected using **training data only**. Test labels are hidden and test inputs are used only after model selection to generate the final prediction files.

## Final models

| Problem | Features | Final model | Selection protocol |
|---|---:|---|---|
| var1 | 6 | Pure Lasso, degree 5, alpha = 0.0070794578 | 5-fold CV repeated 3 times; minimum training-only MSE |
| var2 | 3 | Pure Ridge, degree 11, alpha = 1.5 | 5-fold CV; minimum mean training-only MSE over the refined degree/alpha grid |

Ridge is a pure L2 estimator, so `l1_ratio` is `null` wherever Ridge appears in the result files.

## Repository layout

```text
data/          provided train/test CSVs and sample submission
src/common.py   data loading, polynomial pipelines, training-only CV helpers
scripts/        numbered reproducible pipeline
results/        sweeps, tuning grids, metrics, coefficients, final config
figures/        plots used in the report
models/         fitted sklearn pipelines
predictions/    BT2024008_pred_var1.csv, BT2024008_pred_var2.csv
report/         BT2024008_report.pdf
```

## Included scripts

The numbered scripts contain the analysis, degree-sweep, regularisation-tuning, final-fit/prediction, and feature-ablation workflows used for the submitted results. The final PDF report is supplied directly under `report/`.

## Leakage control

- Test labels are never accessed.
- Test inputs are **not** used for degree selection, regularisation selection, feature ablation, or model-family choice.
- The test inputs are loaded only in the final-fit script after the model configuration is fixed, to produce predictions in sample-submission order.
- Train/test distribution comparisons in the exploratory figure are descriptive only and do not affect model selection.
