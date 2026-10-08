"""Run drop-one-feature ablation with the final model settings (training-only 5x2 CV)."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.common import RESULTS, cv_path, folds, load

cfg = json.loads((RESULTS / "final_config.json").read_text())
cv = folds(5, 2)
rows = []
for var in (1, 2):
    X, y, _Xt, feats = load(var)
    c = cfg[f"var{var}"]
    kind = "enet" if c["kind"] == "enet" else c["kind"]
    full = cv_path(kind, c["degree"], X, y, [c["alpha"]], l1_ratio=c["l1_ratio"], cv=cv).iloc[0]
    rows.append(dict(var=var, dropped="none", cv_mse=full.cv_mse))
    for j, f in enumerate(feats):
        keep = [i for i in range(len(feats)) if i != j]
        r = cv_path(kind, c["degree"], X[:, keep], y, [c["alpha"]], l1_ratio=c["l1_ratio"], cv=cv).iloc[0]
        rows.append(dict(var=var, dropped=f, cv_mse=r.cv_mse))
        print(rows[-1], flush=True)
pd.DataFrame(rows).round(4).to_csv(RESULTS / "feature_ablation.csv", index=False)
