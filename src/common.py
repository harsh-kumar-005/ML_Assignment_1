"""Shared helpers: data loading, polynomial pipelines, validation utilities."""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROLL = "BT2024008"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
MODELS = ROOT / "models"
PREDS = ROOT / "predictions"
SEED = 2024008

warnings.filterwarnings("ignore")


def load(var):
    """Return (X_train, y_train, X_test, feature_names) as numpy arrays."""
    tr = pd.read_csv(DATA / f"{ROLL}_train_var{var}.csv")
    te = pd.read_csv(DATA / f"{ROLL}_test_var{var}.csv")
    feats = [c for c in tr.columns if c != "y"]
    return tr[feats].to_numpy(), tr["y"].to_numpy(), te[feats].to_numpy(), feats


def clip_count(X):
    """Number of coordinates sitting exactly on the +/-1 boundary per row."""
    return (np.abs(X) == 1.0).sum(axis=1)


def make_model(kind, degree, alpha=0.0, l1_ratio=None):
    """Polynomial expansion -> standardisation -> linear estimator."""
    if kind == "ols":
        est = LinearRegression()
    elif kind == "ridge":
        est = Ridge(alpha=alpha)
    elif kind == "lasso":
        est = Lasso(alpha=alpha, max_iter=20000, tol=1e-4)
    elif kind == "enet":
        est = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, tol=1e-4)
    else:
        raise ValueError(kind)
    return Pipeline([
        ("poly", PolynomialFeatures(degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("est", est),
    ])


def folds(n_splits=5, n_repeats=2):
    return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)


def mse(y, p):
    return float(np.mean((y - p) ** 2))


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2))


def cv_path(kind, degree, X, y, alphas, w=None, l1_ratio=None, cv=None):
    """Cross-validate one (kind, degree) over an alpha grid.

    Lasso/ElasticNet reuse the previous solution as a warm start along the
    decreasing alpha grid, so a whole path costs little more than one fit.
    Returns a DataFrame with training-only mean CV MSE.
    """
    cv = cv or folds()
    if kind in ("ols", "ridge"):
        l1_ratio = None
    elif kind == "lasso":
        l1_ratio = 1.0
    elif l1_ratio is None:
        raise ValueError("l1_ratio is required for ElasticNet")
    alphas = sorted(alphas, reverse=True)
    acc = {a: [] for a in alphas}
    for tr_i, va_i in cv.split(X):
        m = make_model(kind, degree, alphas[0], l1_ratio)
        if kind in ("lasso", "enet"):
            m.named_steps["est"].set_params(warm_start=True)
        for a in alphas:
            if kind != "ols":
                m.named_steps["est"].set_params(alpha=a)
            m.fit(X[tr_i], y[tr_i])
            p = m.predict(X[va_i])
            acc[a].append(mse(y[va_i], p))
    rows = [dict(kind=kind, degree=degree, alpha=a, l1_ratio=l1_ratio,
                 cv_mse=np.mean(acc[a]), cv_std=np.std(acc[a])) for a in alphas]
    return pd.DataFrame(rows)
