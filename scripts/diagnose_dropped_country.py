"""Identify which OECD country (or countries) is dropped from the analytic
panel by the complete-case filter, and report which variable(s) caused the
drop.

Replicates the exact filtering logic of ``fit_main_model_correct.py``.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

OUTCOME    = "v2smpolsoc"
COVARIATES = ["log_pop", "urban_pct", "health_exp_gdp", "ethnic_frac"]
NEEDED = [
    OUTCOME,
    "v2smpolsoc_lag1",
    "p_score_mean_lag1",
    "stringency_mean_lag1",
    *COVARIATES,
    "region",
]


def main() -> None:
    df = pd.read_parquet(DATA / "oecd_panel.parquet")
    if "log_pop" not in df.columns:
        df["log_pop"] = np.log(df["population"])
    df = df.sort_values(["iso3", "year"]).set_index(["iso3", "year"])
    df["v2smpolsoc_lag1"]      = df.groupby(level=0)[OUTCOME].shift(1)
    df["p_score_mean_lag1"]    = df.groupby(level=0)["p_score_mean"].shift(1)
    df["stringency_mean_lag1"] = df.groupby(level=0)["stringency_mean"].shift(1)

    all_iso = set(df.index.get_level_values("iso3").unique())
    kept    = set(df.dropna(subset=NEEDED)
                    .index.get_level_values("iso3").unique())
    dropped = sorted(all_iso - kept)

    print(f"[panel] OECD countries in raw panel: {len(all_iso)}")
    print(f"[panel] Countries kept after dropna: {len(kept)}")
    print(f"[panel] DROPPED: {dropped if dropped else 'none'}")

    if not dropped:
        return

    print("\nPer-country NaN audit on the variables required by the model:")
    for iso in dropped:
        sub = df.loc[iso, NEEDED]
        nan_cols = sub.columns[sub.isna().any()].tolist()
        all_nan  = sub.columns[sub.isna().all()].tolist()
        print(f"\n  --- {iso} (rows={len(sub)}) ---")
        print(f"  any-NaN columns: {nan_cols}")
        print(f"  fully-missing columns (NaN every year): {all_nan}")
        print("  NaN count per column:")
        print(sub.isna().sum().to_string().replace("\n", "\n    "))


if __name__ == "__main__":
    main()
