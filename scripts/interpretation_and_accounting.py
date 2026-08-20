"""Interpretation quantities and estimation-sample accounting.

Derives, from the released data and the primary-model posterior, every
revision-added quantity quoted in the manuscript's Results subsection
"Interpretation of parameters", the Methods sample accounting, and SI
Appendix G's residual-autocorrelation diagnostic:

1. Sample accounting: 228 source country-years -> 111 estimation
   country-years (37 countries x 2021-2023), with the excluded blocks
   itemized; lag-2 sample (96) and joint-lag sample (74).
2. Native-scale persistence: rho = phi / SD(lagged outcome), its 95% HDI,
   P(stationary), and the half-life of deviations.
3. Effect-size anchors for the null shock coefficients (SD-unit bounds).
4. Pooled lag-1 autocorrelation of posterior-mean residuals (Keele-Kelly
   precondition check; SI Appendix G).

Requires ``output/idata_main_correct.nc`` — run
``python scripts/fit_main_model_correct.py`` first if it is missing.

Output: output/tables/interpretation_quantities.json (and stdout).
"""
from __future__ import annotations
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
TABLES = OUT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

OUTCOME = "v2smpolsoc"
COVARIATES = ["log_pop", "urban_pct", "health_exp_gdp", "ethnic_frac"]
SEED = 20260503  # fixed across the pipeline


def zscore(s: pd.Series) -> np.ndarray:
    return ((s - s.mean()) / s.std(ddof=0)).to_numpy()


def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "oecd_panel.parquet")
    if "log_pop" not in df.columns:
        df["log_pop"] = np.log(df["population"])
    df = df.sort_values(["iso3", "year"]).set_index(["iso3", "year"])
    for v in [OUTCOME, "p_score_mean", "stringency_mean"]:
        df[f"{v}_lag1"] = df.groupby(level=0)[v].shift(1)
        df[f"{v}_lag2"] = df.groupby(level=0)[v].shift(2)
    return df


def sample_accounting(df: pd.DataFrame) -> dict:
    total = len(df)
    need1 = [OUTCOME, "v2smpolsoc_lag1", "p_score_mean_lag1",
             "stringency_mean_lag1", *COVARIATES, "region"]
    est1 = df.dropna(subset=need1)
    need2 = [OUTCOME, "v2smpolsoc_lag2", "p_score_mean_lag2",
             "stringency_mean_lag2", *COVARIATES, "region"]
    est2 = df.dropna(subset=need2)
    need3 = [OUTCOME, "v2smpolsoc_lag1", "v2smpolsoc_lag2",
             "p_score_mean_lag1", "p_score_mean_lag2",
             "stringency_mean_lag1", "stringency_mean_lag2",
             *COVARIATES, "region"]
    est3 = df.dropna(subset=need3)
    dropped_countries = sorted(set(df.index.get_level_values("iso3"))
                               - set(est1.index.get_level_values("iso3")))
    acc = {
        "source_country_years": int(total),
        "source_countries": int(df.index.get_level_values("iso3").nunique()),
        "lag1_N": int(len(est1)),
        "lag1_countries": int(est1.index.get_level_values("iso3").nunique()),
        "lag1_years": sorted(int(y) for y in set(est1.index.get_level_values("year"))),
        "countries_excluded_entirely": dropped_countries,  # TUR: no OWID excess-mortality series
        "lag2_N": int(len(est2)),
        "joint_lag_N": int(len(est3)),
    }
    assert acc["lag1_N"] == 111 and acc["lag2_N"] == 96 and acc["joint_lag_N"] == 74
    return acc


def main() -> None:
    df = load_panel()
    acc = sample_accounting(df)

    idata_path = OUT / "idata_main_correct.nc"
    if not idata_path.exists():
        raise SystemExit("output/idata_main_correct.nc not found - run "
                         "'python scripts/fit_main_model_correct.py' first.")
    idata = az.from_netcdf(idata_path)
    post = idata.posterior

    need1 = [OUTCOME, "v2smpolsoc_lag1", "p_score_mean_lag1",
             "stringency_mean_lag1", *COVARIATES, "region"]
    est = df.dropna(subset=need1).copy()

    sd_ylag = est["v2smpolsoc_lag1"].std(ddof=0)
    sd_y_cross = est[OUTCOME].std(ddof=0)
    sd_p = est["p_score_mean_lag1"].std(ddof=0)
    sd_s = est["stringency_mean_lag1"].std(ddof=0)

    # --- native-scale persistence, stationarity, half-life -----------------
    phi = post["phi"].values.ravel()
    rho = phi / sd_ylag
    rho_hdi = az.hdi(rho, hdi_prob=0.95)
    half_life_median = float(np.median(np.log(0.5) / np.log(rho[rho < 1])))

    # --- SD-unit bounds for the shock coefficients -------------------------
    def sd_bound(name: str) -> dict:
        b = post[name].values.ravel()
        h = az.hdi(b, hdi_prob=0.95)
        return {"mean": float(b.mean()), "hdi": [float(h[0]), float(h[1])],
                "max_abs_bound_in_outcome_SD": float(max(abs(h[0]), abs(h[1])) / sd_y_cross)}

    # --- residual lag-1 autocorrelation (Keele-Kelly precondition) ---------
    gamma_mean = post["gamma"].mean(dim=("chain", "draw")).values
    a_country = post["a_country"].mean(dim=("chain", "draw")).to_series()
    iso = est.index.get_level_values("iso3")
    mu_hat = (a_country.loc[iso].to_numpy()
              + float(phi.mean()) * zscore(est["v2smpolsoc_lag1"])
              + float(post["beta_p"].values.mean()) * zscore(est["p_score_mean_lag1"])
              + float(post["beta_s"].values.mean()) * zscore(est["stringency_mean_lag1"])
              + np.column_stack([zscore(est[c]) for c in COVARIATES]) @ gamma_mean)
    est["resid"] = est[OUTCOME].to_numpy() - mu_hat
    pairs = np.array([(r[i], r[i + 1])
                      for _, g in est.groupby(level=0)
                      for r in [g["resid"].to_numpy()]
                      for i in range(len(r) - 1)])
    resid_ac1 = float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1])

    result = {
        "sample_accounting": acc,
        "scales": {"sd_lagged_outcome": float(sd_ylag),
                   "sd_outcome_cross_country": float(sd_y_cross),
                   "sd_mortality_lag1_Pscore_points": float(sd_p),
                   "sd_stringency_lag1_index_points": float(sd_s)},
        "native_scale_persistence": {"mean": float(rho.mean()),
                                     "hdi95": [float(rho_hdi[0]), float(rho_hdi[1])],
                                     "P_stationary": float((rho < 1).mean()),
                                     "half_life_years_median": half_life_median},
        "beta_p": sd_bound("beta_p"),
        "beta_s": sd_bound("beta_s"),
        "residual_lag1_autocorrelation": {"value": resid_ac1,
                                          "n_within_country_pairs": int(len(pairs))},
    }
    out_path = TABLES / "interpretation_quantities.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
