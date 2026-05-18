"""Hierarchical Bayesian Dynamic Panel Model (PyMC v5).

Country (iso3) units are partially pooled within UN sub-regions to address
spatial non-independence (Galton's problem) while a within-country
autoregressive term absorbs macro-structural inertia. Every hierarchical
layer is parameterised in non-centred form to avoid Neal's funnel.

The same builder is used for two specifications that differ only in the
lag horizon of the regressors:

* Primary specification (``lag=1``) — Table 1, Figure 1.
* Sensitivity specification (``lag=2``) — Supplementary Table S1.

Model
-----
    y_{c,t} ~ Normal(mu_{c,t}, sigma_y)
    mu_{c,t} = a_country[c]
             + phi    * v2smpolsoc_lag      (autoregressive inertia)
             + beta_p * p_score_mean_lag    (lagged log COVID mortality)
             + beta_s * stringency_mean_lag (lagged OxCGRT stringency)
             + gamma  @ Z                   (4 macro-covariates)

    a_region  = mu_a + sigma_region  * z_region                  (non-centred)
    a_country = a_region[r(c)] + sigma_country * z_country       (non-centred)

Priors (standardised regressor scale):
    mu_a                                  ~ Normal(0, 2)
    sigma_region, sigma_country, sigma_y  ~ HalfNormal(1)
    phi, beta_p, beta_s, gamma            ~ Normal(0, 1)

Sampling: NUTS, 4 chains, tune=2000, draws=2000, target_accept=0.95,
random_seed = 20260503. Convergence is audited at max R-hat <= 1.01 and
minimum bulk ESS > 400.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from .data_prep import COVARIATES, OUTCOME, REGION_COL, zscore

SEED = 20260503
SAMPLE_KWARGS = dict(
    draws=2000,
    tune=2000,
    chains=4,
    target_accept=0.95,
    random_seed=SEED,
    return_inferencedata=True,
    progressbar=True,
)


def build_model(df: pd.DataFrame, lag: Literal[1, 2] = 1) -> pm.Model:
    """Construct the Hierarchical Bayesian Dynamic Panel Model.

    Parameters
    ----------
    df : pd.DataFrame
        Analytic frame produced by ``data_prep.prepare_lag1`` or
        ``data_prep.prepare_lag2``; must be MultiIndexed on (iso3, year).
    lag : {1, 2}
        Lag horizon used for the autoregressive term and the two
        pandemic-exposure regressors.
    """
    countries = df.index.get_level_values("iso3").astype("category")
    country_codes = countries.codes
    country_labels = countries.categories.to_numpy()

    # Map each country to its UN sub-region (the grouping in which countries
    # are partially pooled).
    c2r = (
        df.reset_index()[["iso3", REGION_COL]]
        .drop_duplicates("iso3")
        .set_index("iso3")
        .loc[country_labels, REGION_COL]
        .astype("category")
    )
    region_labels = c2r.cat.categories.to_numpy()
    country_to_region_idx = c2r.cat.codes.to_numpy()

    # Standardise every regressor on the analytic sample so that the
    # Normal(0, 1) priors function as weakly informative shrinkage.
    y = df[OUTCOME].to_numpy()
    x_lag = zscore(df[f"{OUTCOME}_lag{lag}"])
    x_p = zscore(df[f"p_score_mean_lag{lag}"])
    x_s = zscore(df[f"stringency_mean_lag{lag}"])
    Z = np.column_stack([zscore(df[c]) for c in COVARIATES])

    coords = {
        "country": country_labels,
        "region": region_labels,
        "covar": COVARIATES,
        "obs": np.arange(len(df)),
    }

    with pm.Model(coords=coords) as model:
        country_idx = pm.Data("country_idx", country_codes, dims="obs")
        c2r_idx = pm.Data(
            "country_to_region", country_to_region_idx, dims="country"
        )

        # Region intercepts (non-centred).
        mu_a = pm.Normal("mu_a", 0.0, 2.0)
        sigma_region = pm.HalfNormal("sigma_region", 1.0)
        z_region = pm.Normal("z_region", 0.0, 1.0, dims="region")
        a_region = pm.Deterministic(
            "a_region", mu_a + sigma_region * z_region, dims="region"
        )

        # Country intercepts nested in region (non-centred).
        sigma_country = pm.HalfNormal("sigma_country", 1.0)
        z_country = pm.Normal("z_country", 0.0, 1.0, dims="country")
        a_country = pm.Deterministic(
            "a_country",
            a_region[c2r_idx] + sigma_country * z_country,
            dims="country",
        )

        # Scalar coefficients of association.
        phi = pm.Normal("phi", 0.0, 1.0)                  # autoregressive inertia
        beta_p = pm.Normal("beta_p", 0.0, 1.0)            # lagged COVID mortality
        beta_s = pm.Normal("beta_s", 0.0, 1.0)            # lagged stringency
        gamma = pm.Normal("gamma", 0.0, 1.0, dims="covar")  # macro-covariates

        mu = (
            a_country[country_idx]
            + phi * x_lag
            + beta_p * x_p
            + beta_s * x_s
            + pm.math.dot(Z, gamma)
        )
        sigma_y = pm.HalfNormal("sigma_y", 1.0)
        pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y, dims="obs")

    return model


def fit(df: pd.DataFrame, lag: Literal[1, 2] = 1) -> az.InferenceData:
    """Build the model at the requested lag and run NUTS to convergence.

    Returns the resulting :class:`arviz.InferenceData`. Convergence is
    audited at the pre-specified thresholds (max R-hat <= 1.01,
    minimum bulk ESS > 400).
    """
    model = build_model(df, lag=lag)
    with model:
        idata = pm.sample(**SAMPLE_KWARGS)

    diag = az.summary(
        idata,
        var_names=[
            "phi", "beta_p", "beta_s", "gamma",
            "mu_a", "sigma_region", "sigma_country", "sigma_y",
        ],
        hdi_prob=0.95,
    )
    max_rhat = float(diag["r_hat"].max())
    min_ess = float(diag["ess_bulk"].min())
    print(f"[diag lag-{lag}] max R-hat = {max_rhat:.4f}, "
          f"min bulk-ESS = {min_ess:.0f}")
    assert max_rhat <= 1.01, f"R-hat {max_rhat} exceeds 1.01 (lag={lag})"
    assert min_ess > 400.0, f"bulk ESS {min_ess} below 400 (lag={lag})"
    return idata


def save_idata(idata: az.InferenceData, path: Path) -> Path:
    """Persist an InferenceData object to NetCDF and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(path)
    print(f"[saved] {path}")
    return path
