"""Sensitivity Analysis 4 (SI Appendix F) — Alternative outcome (v2cacamps).

Re-estimates the primary Lag-1 Hierarchical Bayesian Dynamic Panel Model
replacing the outcome with V-Dem's political-polarization indicator
(``v2cacamps``: "Is society polarized into antagonistic, political camps?",
0 = not at all ... 4 = to a large extent), a different question answered by a
distinct expert battery. The indicator is reverse-scored (multiplied by -1)
so that, as in the primary outcome, higher values denote weaker polarization.
Specification, priors, sampler settings, and the estimation sample
(111 country-years) are otherwise identical to ``fit_main_model_correct.py``.

Sampling: NUTS, 4 chains, tune=2000, draws=2000, target_accept=0.95.

Outputs
-------
output/idata_sensitivity_altoutcome.nc
output/tables/posterior_summary_si_altoutcome_95hdi.csv   (Supplementary Table S4)
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

COVARIATES = ["log_pop", "urban_pct", "health_exp_gdp", "ethnic_frac"]
REGION_COL = "region"
SEED = 20260503


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "oecd_panel.parquet")
    if "log_pop" not in df.columns:
        df["log_pop"] = np.log(df["population"])
    df = df.sort_values(["iso3", "year"]).set_index(["iso3", "year"])

    alt = (pd.read_csv(DATA / "EIPSA_OECD_panel_2019_2024.csv")
           [["iso3", "year", "v2cacamps"]].set_index(["iso3", "year"]))
    df = df.join(alt, how="left")
    corr = df[["v2smpolsoc", "v2cacamps"]].corr().iloc[0, 1]
    print(f"corr(v2smpolsoc, v2cacamps) = {corr:.3f}  (expected strongly negative)")

    df["coh_alt"] = -df["v2cacamps"]  # reverse-score: higher = weaker polarization
    df["coh_alt_lag1"] = df.groupby(level=0)["coh_alt"].shift(1)
    df["p_score_mean_lag1"] = df.groupby(level=0)["p_score_mean"].shift(1)
    df["stringency_mean_lag1"] = df.groupby(level=0)["stringency_mean"].shift(1)
    needed = ["coh_alt", "coh_alt_lag1", "p_score_mean_lag1",
              "stringency_mean_lag1", *COVARIATES, REGION_COL]
    return df.dropna(subset=needed).copy()


def zscore(s: pd.Series) -> np.ndarray:
    return ((s - s.mean()) / s.std(ddof=0)).to_numpy()


def main() -> None:
    df = load_and_prepare()
    assert len(df) == 111, f"expected 111 country-years, got {len(df)}"

    countries = df.index.get_level_values("iso3").astype("category")
    country_codes = countries.codes
    country_labels = countries.categories.to_numpy()
    c2r = (df.reset_index()[["iso3", REGION_COL]]
             .drop_duplicates("iso3").set_index("iso3")
             .loc[country_labels, REGION_COL].astype("category"))
    region_labels = c2r.cat.categories.to_numpy()
    r_of_c = c2r.cat.codes.to_numpy()

    y = df["coh_alt"].to_numpy(float)
    x_lag = zscore(df["coh_alt_lag1"])
    x_p = zscore(df["p_score_mean_lag1"])
    x_s = zscore(df["stringency_mean_lag1"])
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

        phi = pm.Normal("phi", 0.0, 1.0)
        beta_p = pm.Normal("beta_p", 0.0, 1.0)
        beta_s = pm.Normal("beta_s", 0.0, 1.0)
        gamma = pm.Normal("gamma", 0.0, 1.0, dims="covar")
        sigma_y = pm.HalfNormal("sigma_y", 1.0)

        mu = (a_country[country_codes] + phi * x_lag + beta_p * x_p
              + beta_s * x_s + pm.math.dot(Z, gamma))
        pm.Normal("y", mu=mu, sigma=sigma_y, observed=y)

        idata = pm.sample(draws=2000, tune=2000, chains=4, cores=4,
                          target_accept=0.95, random_seed=SEED)

    idata.to_netcdf(OUT / "idata_sensitivity_altoutcome.nc")
    summary = az.summary(idata, hdi_prob=0.95,
                         var_names=["phi", "beta_p", "beta_s", "gamma",
                                    "mu_a", "sigma_region", "sigma_country", "sigma_y"])
    summary.to_csv(TABLES / "posterior_summary_si_altoutcome_95hdi.csv")
    div = int(idata.sample_stats["diverging"].values.sum())
    print(summary[["mean", "sd", "hdi_2.5%", "hdi_97.5%", "ess_bulk", "r_hat"]].round(3))
    print(f"divergent transitions: {div} / 8000")


if __name__ == "__main__":
    main()
