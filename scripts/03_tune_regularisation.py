"""Fine regularisation search using training data only.

var1: degree fixed at 5; ElasticNet grid over alpha and l1_ratio. l1_ratio=1 is pure Lasso.
var2: Ridge grid over degrees 8-13 and alpha; the degree/alpha pair with the minimum
      mean training-only CV MSE is selected.
"""
import json
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
cfg_path = RESULTS / "final_config.json"
cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

if var == 1:
    cv = folds(n_splits=5, n_repeats=3)
    degree = 5
    alphas = list(np.logspace(-2.6, -1.4, 9))
    l1s = [0.6, 0.8, 0.95, 1.0]
    frames = []
    for l1 in l1s:
        r = cv_path("enet", degree, X, y, alphas, l1_ratio=l1, cv=cv)
        frames.append(r)
        b = r.loc[r.cv_mse.idxmin()]
        print(f"l1_ratio={l1:.2f}: best alpha={b.alpha:.5f} training-only CV={b.cv_mse:.4f}", flush=True)
    grid = pd.concat(frames, ignore_index=True)
    grid.to_csv(RESULTS / "tuning_grid_var1.csv", index=False)
    best = grid.loc[grid.cv_mse.idxmin()]
    cfg["var1"] = dict(kind="lasso" if best.l1_ratio == 1.0 else "enet", degree=degree,
                       alpha=float(best.alpha), l1_ratio=float(best.l1_ratio),
                       cv_mse=float(best.cv_mse), selection_cv="5-fold repeated 3 times; training data only")
    piv = grid.pivot(index="l1_ratio", columns="alpha", values="cv_mse")
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{a:.4f}" for a in piv.columns], rotation=45, fontsize=7)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", fontsize=6, color="white")
    ax.set_xlabel("alpha (regularisation strength)"); ax.set_ylabel("l1_ratio")
    ax.set_title("var1: degree-5 training-only CV MSE")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(FIGS / "tuning_heatmap_var1.png", dpi=150); plt.close(fig)
else:
    # Five-fold training-only tuning. The final model is selected by the minimum
    # mean CV MSE over the refined degree/alpha grid; no test inputs are used.
    cv = folds(n_splits=5, n_repeats=1)
    degrees = [8, 9, 10, 11, 12, 13]
    # Include alpha=1.5 explicitly because it is the final Ridge setting.
    alphas = sorted(set(np.logspace(-1.5, 1.2, 12).tolist() + [1.5]))
    frames = [cv_path("ridge", d, X, y, alphas, cv=cv) for d in degrees]
    grid = pd.concat(frames, ignore_index=True)
    grid.to_csv(RESULTS / "tuning_grid_var2.csv", index=False)
    per_deg = grid.loc[grid.groupby("degree").cv_mse.idxmin()].set_index("degree")
    print(per_deg[["alpha", "cv_mse", "cv_std"]].round(4))
    best_idx = grid.cv_mse.idxmin()
    best = grid.loc[best_idx]
    chosen = int(best.degree)
    print(f"best degree={chosen}, alpha={best.alpha:.6g}, CV MSE={best.cv_mse:.4f}")
    cfg["var2"] = dict(kind="ridge", degree=chosen, alpha=float(best.alpha), l1_ratio=None,
                       cv_mse=float(best.cv_mse), best_degree_by_mse=int(chosen),
                       selection_cv="5-fold; training data only; minimum mean CV MSE")
    piv = grid.pivot(index="degree", columns="alpha", values="cv_mse")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(np.log10(piv.values), aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{a:.2f}" for a in piv.columns], rotation=45, fontsize=7)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", fontsize=5.5, color="white")
    ax.set_xlabel("alpha (ridge penalty)"); ax.set_ylabel("polynomial degree")
    ax.set_title("var2: ridge training-only CV MSE (colour = log10 MSE)")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(FIGS / "tuning_heatmap_var2.png", dpi=150); plt.close(fig)

cfg_path.write_text(json.dumps(cfg, indent=2))
print(json.dumps(cfg[f"var{var}"], indent=2))
