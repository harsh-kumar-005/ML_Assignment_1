"""Fit the selected model on all training rows, report metrics, write predictions.

usage: python scripts/04_fit_predict.py <var>
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from src.common import (DATA, FIGS, MODELS, PREDS, RESULTS, ROLL, SEED, clip_count,
                        folds, load, make_model, r2, mse)

var = int(sys.argv[1])
cfg = json.loads((RESULTS / "final_config.json").read_text())[f"var{var}"]
X, y, Xt, feats = load(var)
build = lambda: make_model(cfg["kind"], cfg["degree"], cfg["alpha"], cfg["l1_ratio"])
metrics = dict(var=var, **{k: cfg[k] for k in ("kind", "degree", "alpha", "l1_ratio")})
metrics["selection_protocol"] = "training-only cross-validation; test inputs used only for final prediction"

# 1) repeated CV on the full training set (out-of-fold predictions)
oof = cross_val_predict(build(), X, y, cv=KFold(5, shuffle=True, random_state=SEED))
metrics["oof_mse"], metrics["oof_r2"] = mse(y, oof), r2(y, oof)
scores = []
for tr_i, va_i in folds(5, 5).split(X):
    m = build().fit(X[tr_i], y[tr_i]); p = m.predict(X[va_i])
    scores.append((mse(y[va_i], p), r2(y[va_i], p)))
metrics["cv5x5_mse"], metrics["cv5x5_r2"] = np.mean(scores, 0).tolist()
metrics["cv5x5_mse_std"] = float(np.std([s[0] for s in scores]))

# 2) single 80/20 holdout
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=SEED)
p = build().fit(Xa, ya).predict(Xb)
metrics["holdout_mse"], metrics["holdout_r2"] = mse(yb, p), r2(yb, p)

# 3) shift stress test (var1 only): learn on rows with few clipped features,
#    evaluate on rows with many clipped features, like the real test set
if var == 1:
    k = clip_count(X)
    lo, hi = k <= 3, k >= 4
    p = build().fit(X[lo], y[lo]).predict(X[hi])
    metrics["shift_train_rows"], metrics["shift_eval_rows"] = int(lo.sum()), int(hi.sum())
    metrics["shift_mse"], metrics["shift_r2"] = mse(y[hi], p), r2(y[hi], p)
    # Boundary stress test uses training rows only; test inputs are not used for selection or scoring.

# 4) final fit on everything
model = build().fit(X, y)
ptr = model.predict(X)
metrics["train_mse"], metrics["train_r2"] = mse(y, ptr), r2(y, ptr)
est = model.named_steps["est"]
metrics["n_terms"] = int(est.coef_.size)
metrics["n_nonzero_terms"] = int(np.sum(np.abs(est.coef_) > 1e-12))

pred = model.predict(Xt)
metrics["test_pred_min"], metrics["test_pred_max"] = float(pred.min()), float(pred.max())
metrics["test_pred_mean"], metrics["test_pred_std"] = float(pred.mean()), float(pred.std())
metrics["train_y_min"], metrics["train_y_max"] = float(y.min()), float(y.max())

# artefacts ------------------------------------------------------------------
joblib.dump(model, MODELS / f"polynomial_model_var{var}.joblib")
names = model.named_steps["poly"].get_feature_names_out(feats)
coef = pd.DataFrame(dict(term=names, standardised_coef=est.coef_,
                         feature_mean=model.named_steps["scale"].mean_,
                         feature_scale=model.named_steps["scale"].scale_))
coef.loc[len(coef)] = ["(intercept)", est.intercept_, np.nan, np.nan]
coef["abs"] = coef.standardised_coef.abs()
coef.sort_values("abs", ascending=False).drop(columns="abs").to_csv(
    RESULTS / f"coefficients_var{var}.csv", index=False)

sample = pd.read_csv(DATA / "sample_submission.csv")
assert len(pred) == len(sample) == len(Xt) and np.isfinite(pred).all()
pd.DataFrame({"y": pred}).to_csv(PREDS / f"{ROLL}_pred_var{var}.csv", index=False)

(RESULTS / f"metrics_var{var}.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))

# figures ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].scatter(y, oof, s=6, alpha=0.5, color="#4c78a8")
lim = [min(y.min(), oof.min()), max(y.max(), oof.max())]
ax[0].plot(lim, lim, "r--", lw=1)
ax[0].set_xlabel("actual y"); ax[0].set_ylabel("out-of-fold prediction")
ax[0].set_title(f"var{var}: OOF R2 = {metrics['oof_r2']:.3f}")
res = y - oof
ax[1].scatter(oof, res, s=6, alpha=0.5, color="#54a24b"); ax[1].axhline(0, color="r", lw=1)
ax[1].set_xlabel("out-of-fold prediction"); ax[1].set_ylabel("residual"); ax[1].set_title("residuals")
ax[2].hist(res, bins=40, color="#f58518", edgecolor="white")
ax[2].set_xlabel("residual"); ax[2].set_title(f"residual std = {res.std():.3f}")
fig.tight_layout(); fig.savefig(FIGS / f"fit_diagnostics_var{var}.png", dpi=150); plt.close(fig)

if var == 1:
    k = clip_count(X)
    levels = sorted(set(k))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.boxplot([np.abs(res[k == j]) for j in levels], labels=levels, showfliers=False)
    ax.set_xlabel("number of clipped features in row"); ax.set_ylabel("|OOF residual|")
    ax.set_title("var1: error against boundary proximity")
    fig.tight_layout(); fig.savefig(FIGS / "residual_by_clipcount_var1.png", dpi=150); plt.close(fig)

# coefficient magnitudes
top = coef.iloc[:-1].assign(abs=lambda d: d.standardised_coef.abs()).nlargest(15, "abs")
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(top.term[::-1], top.standardised_coef[::-1], color="#4c78a8")
ax.set_title(f"var{var}: 15 largest standardised coefficients")
ax.tick_params(axis="y", labelsize=7)
fig.tight_layout(); fig.savefig(FIGS / f"top_coefficients_var{var}.png", dpi=150); plt.close(fig)
