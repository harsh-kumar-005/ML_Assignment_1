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

## Reproducibility

All selection results are generated from fixed training-only CV seeds. The submitted model artifacts, coefficient tables, prediction files, validation tables, and PDF report are included together and use the same final configuration.

GitHub: add your NEW repository URL here after creating the repository.

## New GitHub Repository: Step-by-Step Commit

This submission includes `commit.sh` as a guided first-time Git setup. It intentionally **does not perform all Git actions at once**. Each step pauses so you can review the result before continuing.

### 1. Create a new EMPTY GitHub repository

Create the repository on GitHub without adding a README, `.gitignore`, or license. This avoids conflicts with the local submission files.

### 2. Open a terminal in this submission folder

```bash
cd ML_Assignment1
```

### 3. Make the script executable

```bash
chmod +x commit.sh
```

### 4. Run the guided commit script

```bash
./commit.sh
```

The script walks through these checkpoints:

1. Check Git installation
2. Initialize the local repository with `git init`
3. Set the branch to `main`
4. Add the URL of your new GitHub repository as `origin`
5. Review `git status`
6. Stage files with `git add -A`
7. Review the staged files and staged diff summary
8. Create the first commit: `Initial submission - ML Assignment 1`
9. Verify the commit with `git status` and `git log -1`
10. Push with `git push -u origin main`

The script asks for confirmation before staging, committing, and pushing. If you stop before pushing, the commit remains safely on your local machine and you can push later with:

```bash
git push -u origin main
```

### Manual commands (if you prefer not to use the script)

```bash
git init
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git status
git add -A
git status --short
git diff --cached --stat
git commit -m "Initial submission - ML Assignment 1"
git status
git log -1 --oneline
git push -u origin main
```
