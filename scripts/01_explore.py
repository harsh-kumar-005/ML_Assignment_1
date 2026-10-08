"""Exploratory data analysis: summary tables and distribution plots."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import FIGS, RESULTS, clip_count, load

rows = []
for var in (1, 2):
    X, y, Xt, feats = load(var)
    for j, f in enumerate(feats):
        rows.append(dict(var=var, feature=f,
                         train_mean=X[:, j].mean(), train_std=X[:, j].std(),
                         test_mean=Xt[:, j].mean(), test_std=Xt[:, j].std(),
                         train_frac_clipped=np.mean(np.abs(X[:, j]) == 1),
                         test_frac_clipped=np.mean(np.abs(Xt[:, j]) == 1),
                         corr_with_y=np.corrcoef(X[:, j], y)[0, 1]))
    print(f"var{var}: n_train={len(y)}, n_test={len(Xt)}, y mean={y.mean():.3f}, y std={y.std():.3f}, "
          f"range=({y.min():.2f}, {y.max():.2f})")
pd.DataFrame(rows).round(4).to_csv(RESULTS / "eda_feature_summary.csv", index=False)

# --- figure 1: target distributions --------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for a, var in zip(ax, (1, 2)):
    _, y, _, _ = load(var)
    a.hist(y, bins=40, color="#4c78a8", edgecolor="white")
    a.set_title(f"var{var}: distribution of y")
    a.set_xlabel("y"); a.set_ylabel("count")
fig.tight_layout(); fig.savefig(FIGS / "eda_target_hist.png", dpi=150); plt.close(fig)

# --- figure 2: train vs test boundary clipping ----------------------------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for a, var in zip(ax, (1, 2)):
    X, _, Xt, feats = load(var)
    k = np.arange(len(feats) + 1)
    p_tr = np.bincount(clip_count(X), minlength=len(k)) / len(X)
    p_te = np.bincount(clip_count(Xt), minlength=len(k)) / len(Xt)
    a.bar(k - 0.2, p_tr, 0.4, label="train", color="#4c78a8")
    a.bar(k + 0.2, p_te, 0.4, label="test", color="#f58518")
    a.set_title(f"var{var}: features sitting at +/-1 per row")
    a.set_xlabel("number of clipped features"); a.set_ylabel("fraction of rows")
    a.legend()
fig.tight_layout(); fig.savefig(FIGS / "eda_clipping_shift.png", dpi=150); plt.close(fig)

# --- figure 3: marginal feature histograms, train vs test ------------------
for var in (1, 2):
    X, _, Xt, feats = load(var)
    n = len(feats)
    fig, ax = plt.subplots(1, n, figsize=(2.6 * n, 2.6), sharey=True)
    for j, a in enumerate(np.atleast_1d(ax)):
        a.hist(X[:, j], bins=25, alpha=0.65, density=True, label="train", color="#4c78a8")
        a.hist(Xt[:, j], bins=25, alpha=0.55, density=True, label="test", color="#f58518")
        a.set_title(feats[j])
    np.atleast_1d(ax)[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / f"eda_features_var{var}.png", dpi=150); plt.close(fig)
print("saved EDA tables and figures")
