"""Sensitivity Analysis 5 (SI Appendix G) — Joint distributed-lag specification.

Extends the primary linear predictor to include lag-1 and lag-2 terms
simultaneously for the outcome and for both pandemic-shock exposures
(six dynamic terms in total), retaining the hierarchical structure, priors,
and sampler settings of ``fit_main_model_correct.py``. Requiring both lags
restricts the estimation sample to 74 country-years (37 countries; outcome
years 2022-2023).

Also reports the derived posterior sums (cumulative two-year associations
and total persistence phi1 + phi2) quoted in Supplementary Table S5.

Sampling: NUTS, 4 chains, tune=2000, draws=2000, target_accept=0.95.

Outputs
-------
output/idata_sensitivity_distlag.nc
output/tables/posterior_summary_si_distlag_95hdi.csv   (Supplementary Table S5)
"""
from __future__ import annotations
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
TABLES = OUT / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

OUTCOME = "v2smpolsoc"
COVARIATES = ["log_pop", "urban_pct", "health_exp_gdp", "ethnic_frac"]
REGION_COL = "region"
SEED = 20260503

DYNAMIC_TERMS = ["phi1", "phi2", "beta_p1", "beta_p2", "beta_s1", "beta_s2"]


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "oecd_panel.parquet")
    if "log_pop" not in df.columns:
        df["log_pop"] = np.log(df["population"])
    df = df.sort_values(["iso3", "year"]).set_index(["iso3", "year"])
    for v in [OUTCOME, "p_score_mean", "stringency_mean"]:
        df[f"{v}_lag1"] = df.groupby(level=0)[v].shift(1)
        df[f"{v}_lag2"] = df.groupby(level=0)[v].shift(2)
    needed = [OUTCOME,
              "v2smpolsoc_lag1", "v2smpolsoc_lag2",
              "p_score_mean_lag1", "p_score_mean_lag2",
              "stringency_mean_lag1", "stringency_mean_lag2",
              *COVARIATES, REGION_COL]
    return df.dropna(subset=needed).copy()


def zscore(s: pd.Series) -> np.ndarray:
    return ((s - s.mean()) / s.std(ddof=0)).to_numpy()


def main() -> None:
    df = load_and_prepare()
    assert len(df) == 74, f"expected 74 country-years, got {len(df)}"

    countries = df.index.get_level_values("iso3").astype("category")
    country_codes = countries.codes
    country_labels = countries.categories.to_numpy()
    c2r = (df.reset_index()[["iso3", REGION_COL]]
             .drop_duplicates("iso3").set_index("iso3")
             .loc[country_labels, REGION_COL].astype("category"))
    region_labels = c2r.cat.categories.to_numpy()
    r_of_c = c2r.cat.codes.to_numpy()

    y = df[OUTCOME].to_numpy(float)
    X = {
        "phi1": zscore(df["v2smpolsoc_lag1"]),
        "phi2": zscore(df["v2smpolsoc_lag2"]),
        "beta_p1": zscore(df["p_score_mean_lag1"]),
        "beta_p2": zscore(df["p_score_mean_lag2"]),
        "beta_s1": zscore(df["stringency_mean_lag1"]),
        "beta_s2": zscore(df["stringency_mean_lag2"]),
    }
    Z = np.column_stack([zscore(df[c]) for c in COVARIATES])

    coords = {"country": country_labels, "region": region_labels, "covar": COVARIATES}
    with pm.Model(coords=coords):
        mu_a = pm.Normal("mu_a", 0.0, 2.0)
        sigma_region = pm.HalfNormal("sigma_region", 1.0)
        sigma_country = pm.HalfNormal("sigma_country", 1.0)
        z_region = pm.Normal("z_region", 0.0, 1.0, dims="region")
        z_country = pm.Normal("z_country", 0.0, 1.0, dims="country")
        a_region = pm.Deterministic("a_region", mu_a + sigma_region * z_region, dims="region")
        a_country = pm.Deterministic(
            "a_country", a_region[r_of_c] + sigma_country * z_country, dims="country")

        betas = {name: pm.Normal(name, 0.0, 1.0) for name in DYNAMIC_TERMS}
        gamma = pm.Normal("gamma", 0.0, 1.0, dims="covar")
        sigma_y = pm.HalfNormal("sigma_y", 1.0)

        mu = (a_country[country_codes]
              + sum(betas[name] * X[name] for name in DYNAMIC_TERMS)
              + pm.math.dot(Z, gamma))
        pm.Normal("y", mu=mu, sigma=sigma_y, observed=y)

        # derived posterior sums (cumulative associations; total persistence)
        pm.Deterministic("cum_mortality", betas["beta_p1"] + betas["beta_p2"])
        pm.Deterministic("cum_stringency", betas["beta_s1"] + betas["beta_s2"])
        pm.Deterministic("total_persistence", betas["phi1"] + betas["phi2"])

        idata = pm.sample(draws=2000, tune=2000, chains=4, cores=4,
                          target_accept=0.95, random_seed=SEED)

    idata.to_netcdf(OUT / "idata_sensitivity_distlag.nc")
    summary = az.summary(
        idata, hdi_prob=0.95,
        var_names=[*DYNAMIC_TERMS, "cum_mortality", "cum_stringency",
                   "total_persistence", "gamma", "mu_a",
                   "sigma_region", "sigma_country", "sigma_y"])
    summary.to_csv(TABLES / "posterior_summary_si_distlag_95hdi.csv")
    div = int(idata.sample_stats["diverging"].values.sum())
    print(summary[["mean", "sd", "hdi_2.5%", "hdi_97.5%", "ess_bulk", "r_hat"]].round(3))
    print(f"divergent transitions: {div} / 8000")


if __name__ == "__main__":
    main()
