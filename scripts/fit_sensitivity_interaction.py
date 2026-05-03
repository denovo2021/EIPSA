"""Fit the effect-modification (interaction) sensitivity specification of the
Hierarchical Bayesian Dynamic Panel Model and save the trace to
``output/idata_sensitivity_interaction.nc``.

Extends the primary Lag-1 model with two product (interaction) terms between
each one-year-lagged pandemic-shock exposure and a time-invariant baseline
covariate B_c defined as country c's value of v2smpolsoc in 2019. The main
term in B_c is absorbed by the partially pooled country-level intercept under
the strictly non-centered hierarchy; the two product terms are identified
because the underlying lagged exposures vary within country.
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
    df = df.sort_values(["iso3", "year"])

    # Time-invariant baseline cohesion (V-Dem v2smpolsoc in 2019).
    baseline_2019 = (
        df.loc[df["year"] == 2019, ["iso3", OUTCOME]]
          .rename(columns={OUTCOME: "v2smpolsoc_2019"})
    )
    df = df.merge(baseline_2019, on="iso3", how="left")

    df = df.set_index(["iso3", "year"])
    df["v2smpolsoc_lag1"]      = df.groupby(level=0)[OUTCOME].shift(1)
    df["p_score_mean_lag1"]    = df.groupby(level=0)["p_score_mean"].shift(1)
    df["stringency_mean_lag1"] = df.groupby(level=0)["stringency_mean"].shift(1)

    needed = [OUTCOME, "v2smpolsoc_lag1", "p_score_mean_lag1",
              "stringency_mean_lag1", "v2smpolsoc_2019",
              *COVARIATES, REGION_COL]
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
    x_lag = zscore(df["v2smpolsoc_lag1"])
    x_p   = zscore(df["p_score_mean_lag1"])
    x_s   = zscore(df["stringency_mean_lag1"])
    x_b   = zscore(df["v2smpolsoc_2019"])
    Z     = np.column_stack([zscore(df[c]) for c in COVARIATES])

    # Standardized product terms (constructed on the standardized scale so that
    # Normal(0,1) priors on the interaction coefficients are weakly informative
    # in comparable units to the lower-order coefficients).
    x_pb = x_p * x_b
    x_sb = x_s * x_b

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

        phi     = pm.Normal("phi",     0.0, 1.0)
        beta_p  = pm.Normal("beta_p",  0.0, 1.0)
        beta_s  = pm.Normal("beta_s",  0.0, 1.0)
        delta_p = pm.Normal("delta_p", 0.0, 1.0)   # mortality x baseline
        delta_s = pm.Normal("delta_s", 0.0, 1.0)   # stringency x baseline
        gamma   = pm.Normal("gamma",   0.0, 1.0, dims="covar")

        mu = (
            a_country[country_idx]
            + phi     * x_lag
            + beta_p  * x_p
            + beta_s  * x_s
            + delta_p * x_pb
            + delta_s * x_sb
            + pm.math.dot(Z, gamma)
        )
        sigma_y = pm.HalfNormal("sigma_y", 1.0)
        pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y, dims="obs")

    return model


def main() -> None:
    df = load_and_prepare()
    print(f"[panel-IX] N={len(df)} obs, "
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
        var_names=["phi", "beta_p", "beta_s", "delta_p", "delta_s", "gamma",
                   "mu_a", "sigma_region", "sigma_country", "sigma_y"],
        hdi_prob=0.95,
    )
    print(f"[diag] max R-hat = {summary['r_hat'].max():.4f}, "
          f"min bulk-ESS = {summary['ess_bulk'].min():.0f}")
    assert summary["r_hat"].max()    <= 1.01
    assert summary["ess_bulk"].min() >  400.0

    out_path = OUT / "idata_sensitivity_interaction.nc"
    idata.to_netcdf(out_path)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
