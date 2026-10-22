"""Head-to-head Prophet vs Timestral on the fev-bench tasks cached in timestral_study/runs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import fev
from fev.analysis import TASK_DEF_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "timestral_study"
FEV_LB_KW = dict(missing_strategy="impute", baseline_model="Seasonal Naive", leakage_imputation_model="Chronos-Bolt")


def load():
    fev_all = pd.concat([pd.read_csv(f) for f in sorted((STUDY / "fev/benchmarks/fev_bench/results").glob("*.csv"))],
                        ignore_index=True)
    rows = pd.DataFrame([json.loads(p.read_text()) for p in (STUDY / "runs").glob("Prophet_*.json")])
    ref = fev_all.drop_duplicates("task_name").set_index("task_name")
    for c in TASK_DEF_COLUMNS:
        if c in ref and c != "task_name":
            rows[c] = rows["task_name"].map(ref[c])
    names = sorted(rows.task_name.unique())
    combined = pd.concat([fev_all[fev_all.task_name.isin(names)], rows], ignore_index=True)
    return fev_all, combined, names


def leaderboard(df, metric):
    return fev.leaderboard(df, metric_column=metric, n_resamples=1000, **FEV_LB_KW)


if __name__ == "__main__":
    fev_all, combined, names = load()
    print(len(names), "tasks;", combined.model_name.nunique(), "models")
    lb = leaderboard(combined, "SQL")
    print(lb.columns.tolist())
    pd.set_option("display.width", 250)
    print(lb[[c for c in lb.columns if c.startswith(("skill", "win")) or "time" in c]].round(3).to_string())
