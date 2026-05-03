"""Fit the Lag-2 sensitivity specification of the Hierarchical Bayesian
Dynamic Panel Model and save the trace to ``output/idata_sensitivity_lag2.nc``.

Differs from the primary model only in the lag horizon: the autoregressive
term and both pandemic-shock exposures are shifted to a two-year within-
country lag. Hierarchy, priors, sampler settings, and convergence thresholds
are inherited unchanged from the primary specification.
"""
from __future__ import annotations
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT  = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

OUTCOME    = "v2smpolsoc"
COVARIATES = ["log_pop", "urban_pct", "health_exp_gdp", "ethnic_frac"]
REGION_COL = "region"
SEED       = 20260503


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "oecd_panel.parquet")
    if "log_pop" not in df.columns:
        df["log_pop"] = np.log(df["population"])
    df = df.sort_values(["iso3", "year"]).set_index(["iso3", "year"])
    df["v2smpolsoc_lag2"]    = df.groupby(level=0)[OUTCOME].shift(2)
    df["p_score_mean_lag2"]  = df.groupby(level=0)["p_score_mean"].shift(2)
    df["stringency_mean_lag2"] = df.groupby(level=0)["stringency_mean"].shift(2)
    needed = [OUTCOME, "v2smpolsoc_lag2", "p_score_mean_lag2",
              "stringency_mean_lag2", *COVARIATES, REGION_COL]
    return df.dropna(subset=needed).copy()


def zscore(s: pd.Series) -> np.ndarray:
    return ((s - s.mean()) / s.std(ddof=0)).to_numpy()


def build_model(df: pd.DataFrame) -> pm.Model:
    countries = df.index.get_level_values("iso3").astype("category")
    country_codes  = countries.codes
    country_labels = countries.categories.to_numpy()

    c2r = (df.reset_index()[["iso3", REGION_COL]]
             .drop_duplicates("iso3").set_index("iso3")
             .loc[country_labels, REGION_COL].astype("category"))
    region_labels         = c2r.cat.categories.to_numpy()
    country_to_region_idx = c2r.cat.codes.to_numpy()

    y     = df[OUTCOME].to_numpy()
    x_lag = zscore(df["v2smpolsoc_lag2"])
    x_p   = zscore(df["p_score_mean_lag2"])
    x_s   = zscore(df["stringency_mean_lag2"])
    Z     = np.column_stack([zscore(df[c]) for c in COVARIATES])

    coords = {
        "country": country_labels,
        "region":  region_labels,
        "covar":   COVARIATES,
        "obs":     np.arange(len(df)),
    }

    with pm.Model(coords=coords) as model:
        country_idx = pm.Data("country_idx", country_codes, dims="obs")
        c2r_idx     = pm.Data("country_to_region", country_to_region_idx,
                              dims="country")

        mu_a         = pm.Normal("mu_a", 0.0, 2.0)
        sigma_region = pm.HalfNormal("sigma_region", 1.0)
        z_region     = pm.Normal("z_region", 0.0, 1.0, dims="region")
        a_region     = pm.Deterministic(
            "a_region", mu_a + sigma_region * z_region, dims="region"
        )

        sigma_country = pm.HalfNormal("sigma_country", 1.0)
        z_country     = pm.Normal("z_country", 0.0, 1.0, dims="country")
        a_country     = pm.Deterministic(
            "a_country",
            a_region[c2r_idx] + sigma_country * z_country,
            dims="country",
        )

        phi    = pm.Normal("phi",    0.0, 1.0)
        beta_p = pm.Normal("beta_p", 0.0, 1.0)
        beta_s = pm.Normal("beta_s", 0.0, 1.0)
        gamma  = pm.Normal("gamma",  0.0, 1.0, dims="covar")

        mu = (
            a_country[country_idx]
            + phi    * x_lag
            + beta_p * x_p
            + beta_s * x_s
            + pm.math.dot(Z, gamma)
        )
        sigma_y = pm.HalfNormal("sigma_y", 1.0)
        pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y, dims="obs")

    return model


def main() -> None:
    df = load_and_prepare()
    print(f"[panel-L2] N={len(df)} obs, "
          f"{df.index.get_level_values('iso3').nunique()} countries, "
          f"{df[REGION_COL].nunique()} UN sub-regions.")

    model = build_model(df)
    with model:
        idata = pm.sample(
            draws=2000, tune=2000, chains=4,
            target_accept=0.95, random_seed=SEED,
            return_inferencedata=True, progressbar=True,
        )

    summary = az.summary(
        idata,
        var_names=["phi", "beta_p", "beta_s", "gamma",
                   "mu_a", "sigma_region", "sigma_country", "sigma_y"],
        hdi_prob=0.95,
    )
    print(f"[diag] max R-hat = {summary['r_hat'].max():.4f}, "
          f"min bulk-ESS = {summary['ess_bulk'].min():.0f}")
    assert summary["r_hat"].max()    <= 1.01
    assert summary["ess_bulk"].min() >  400.0

    out_path = OUT / "idata_sensitivity_lag2.nc"
    idata.to_netcdf(out_path)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
