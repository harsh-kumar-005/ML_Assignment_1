"""Cross-validated sweep over polynomial degree for OLS, Ridge and Lasso.

usage: python scripts/02_degree_sweep.py <var>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import FIGS, RESULTS, cv_path, folds, load

var = int(sys.argv[1])
X, y, _Xt, _ = load(var)
cv = folds(n_splits=5, n_repeats=1)

if var == 1:
    degrees = range(1, 7)
    grids = {"ridge": np.logspace(-2, 3, 8), "lasso": np.logspace(-3, -0.5, 8)}
    lasso_max = 6
else:
    degrees = range(1, 9)
    grids = {"ridge": np.logspace(-3, 1.5, 8), "lasso": np.logspace(-4.5, -1.5, 8)}
    lasso_max = 8

out = []
for d in degrees:
    cur = [
        # plain least squares reference (no penalty)
        cv_path("ols", d, X, y, [0.0], cv=cv),
        cv_path("ridge", d, X, y, list(grids["ridge"]), cv=cv),
    ]
    if d <= lasso_max:
        cur.append(cv_path("lasso", d, X, y, list(grids["lasso"]), cv=cv))
    out.extend(cur)
    best = pd.concat(cur).sort_values("cv_mse").iloc[0]
    print(f"var{var} degree {d:2d}: best={best.kind:5s} alpha={best.alpha:.4g} cv_mse={best.cv_mse:.4f}", flush=True)
    pd.concat(out).to_csv(RESULTS / f"degree_sweep_var{var}.csv", index=False)

res = pd.concat(out)
best = res.groupby(["kind", "degree"], as_index=False)[["cv_mse"]].min()

fig, ax = plt.subplots(figsize=(6.4, 4))
for kind, c in (("ols", "#e45756"), ("ridge", "#4c78a8"), ("lasso", "#54a24b")):
    s = best[best.kind == kind]
    ax.plot(s.degree, s.cv_mse, "o-", label=kind.upper() if kind == "ols" else kind.capitalize(), color=c)
ax.set_yscale("log")
ax.set_xlabel("polynomial degree"); ax.set_ylabel("5-fold CV MSE (best alpha)")
ax.set_title(f"var{var}: error against polynomial degree")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(FIGS / f"degree_error_var{var}.png", dpi=150); plt.close(fig)
