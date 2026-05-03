"""
Phase 2 (Bayesian, dynamic): hierarchical Bayesian dynamic panel model that
formally evaluates the association between pandemic-related stressors and
social polarization (v2smpolsoc), conditional on macro-structural inertia
(autoregressive lagged outcome) and accounting for spatial non-independence
(Galton's problem) via partial pooling of country-level intercepts within
UN sub-regions.

Hierarchy
---------
    Country (iso3)  nested in  UN sub-region  (column: ``region``)

Model (all hierarchical layers in NON-CENTERED form to avoid Neal's funnel)
---------------------------------------------------------------------------
    z_region   ~ Normal(0, 1)
    a_region   = mu_a + sigma_region * z_region

    z_country  ~ Normal(0, 1)
    a_country  = a_region[r(c)] + sigma_country * z_country

    mu_{c,t}   = a_country[c]
               + phi      * v2smpolsoc_lag1_{c,t}        # macro-structural inertia
               + beta_p   * p_score_mean_lag1_{c,t}      # acute mortality association
               + beta_s   * stringency_mean_lag1_{c,t}   # acute lockdown association
               + beta_pop * log_pop_{c,t}
               + beta_urb * urban_pct_{c,t}
               + beta_hlth* health_exp_gdp_{c,t}
               + beta_eth * ethnic_frac_{c,t}

    y_{c,t}    ~ Normal(mu_{c,t}, sigma_y)

Priors (weakly informative on the standardized regressor scale):
    mu_a                                 ~ Normal(0, 2)
    sigma_region, sigma_country, sigma_y ~ HalfNormal(1)
    phi, beta_*                          ~ Normal(0, 1)

Sampling: NUTS, draws=2000, tune=2000, chains=4, target_accept=0.95.

Run from the project root:
    python scripts/phase2_bayesian_dynamic.py
"""
from __future__ import annotations
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TABLES = ROOT / "output" / "tables"
FIGS = ROOT / "output" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

OUTCOME = "v2smpolsoc"
PRIMARY_EXPOSURES = ["p_score_mean", "stringency_mean"]
COVARIATES = ["log_pop", "urban_pct", "health_exp_gdp", "ethnic_frac"]
REGION_COL = "region"  # UN sub-region label (analogue of ``un_region_name``)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_and_prepare() -> pd.DataFrame:
    """Load the OECD panel, derive lagged covariates, and return the analytic
    frame restricted to complete cases on the variables used by the model."""
    df = pd.read_parquet(DATA / "oecd_panel.parquet")

    # Ensure the log-population covariate is available.
    if "log_pop" not in df.columns:
        df["log_pop"] = np.log(df["population"])

    # Sort and index by (country, year) before computing within-country lags.
    df = df.sort_values(["iso3", "year"]).set_index(["iso3", "year"])

    # 1-year within-country lags for the outcome and the primary exposures.
    df["v2smpolsoc_lag1"] = df.groupby(level=0)[OUTCOME].shift(1)
    df["p_score_mean_lag1"] = df.groupby(level=0)["p_score_mean"].shift(1)
    df["stringency_mean_lag1"] = df.groupby(level=0)["stringency_mean"].shift(1)

    # Restrict to complete cases on the variables that enter the dynamic model.
    needed = [
        OUTCOME,
        "v2smpolsoc_lag1",
        "p_score_mean_lag1",
        "stringency_mean_lag1",
        *COVARIATES,
        REGION_COL,
    ]
    df = df.dropna(subset=needed).copy()

    return df


def standardize(s: pd.Series) -> pd.Series:
    """Z-standardize a series (mean 0, sd 1) so that Normal(0, 1) priors on the
    coefficients are weakly informative on a comparable scale."""
    return (s - s.mean()) / s.std(ddof=0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(df: pd.DataFrame) -> tuple[pm.Model, dict]:
    """Construct the hierarchical Bayesian dynamic panel model."""
    # --- Hierarchy indices: countries nested within UN sub-regions ----------
    countries = df.index.get_level_values("iso3").astype("category")
    country_codes = countries.codes
    country_labels = countries.categories.to_numpy()

    # One region per country (time-invariant by construction).
    country_to_region_label = (
        df.reset_index()[["iso3", REGION_COL]]
        .drop_duplicates("iso3")
        .set_index("iso3")
        .loc[country_labels, REGION_COL]
    )
    regions = country_to_region_label.astype("category")
    region_labels = regions.cat.categories.to_numpy()
    country_to_region_idx = regions.cat.codes.to_numpy()

    n_country = len(country_labels)
    n_region = len(region_labels)

    # --- Standardized design matrix (improves prior calibration) ------------
    y = df[OUTCOME].to_numpy()
    x_lag = standardize(df["v2smpolsoc_lag1"]).to_numpy()
    x_p = standardize(df["p_score_mean_lag1"]).to_numpy()
    x_s = standardize(df["stringency_mean_lag1"]).to_numpy()
    x_pop = standardize(df["log_pop"]).to_numpy()
    x_urb = standardize(df["urban_pct"]).to_numpy()
    x_hlth = standardize(df["health_exp_gdp"]).to_numpy()
    x_eth = standardize(df["ethnic_frac"]).to_numpy()

    coords = {
        "country": country_labels,
        "region": region_labels,
        "obs": np.arange(len(df)),
    }

    with pm.Model(coords=coords) as model:
        # Observed-data containers (kept as Data so the model is portable).
        country_idx = pm.Data("country_idx", country_codes, dims="obs")
        c2r_idx = pm.Data("country_to_region", country_to_region_idx, dims="country")

        # ---- Region-level intercepts (non-centered) ------------------------
        mu_a = pm.Normal("mu_a", mu=0.0, sigma=2.0)
        sigma_region = pm.HalfNormal("sigma_region", sigma=1.0)
        z_region = pm.Normal("z_region", mu=0.0, sigma=1.0, dims="region")
        a_region = pm.Deterministic(
            "a_region", mu_a + z_region * sigma_region, dims="region"
        )

        # ---- Country-level intercepts nested in region (non-centered) -----
        sigma_country = pm.HalfNormal("sigma_country", sigma=1.0)
        z_country = pm.Normal("z_country", mu=0.0, sigma=1.0, dims="country")
        a_country = pm.Deterministic(
            "a_country",
            a_region[c2r_idx] + z_country * sigma_country,
            dims="country",
        )

        # ---- Coefficients of association ----------------------------------
        # Macro-structural inertia (autoregressive association).
        phi = pm.Normal("phi", mu=0.0, sigma=1.0)
        # Acute pandemic-shock associations (lagged exposures).
        beta_p = pm.Normal("beta_p", mu=0.0, sigma=1.0)
        beta_s = pm.Normal("beta_s", mu=0.0, sigma=1.0)
        # Macro-covariate associations.
        beta_pop = pm.Normal("beta_pop", mu=0.0, sigma=1.0)
        beta_urb = pm.Normal("beta_urb", mu=0.0, sigma=1.0)
        beta_hlth = pm.Normal("beta_hlth", mu=0.0, sigma=1.0)
        beta_eth = pm.Normal("beta_eth", mu=0.0, sigma=1.0)

        # ---- Linear predictor ---------------------------------------------
        mu = (
            a_country[country_idx]
            + phi * x_lag
            + beta_p * x_p
            + beta_s * x_s
            + beta_pop * x_pop
            + beta_urb * x_urb
            + beta_hlth * x_hlth
            + beta_eth * x_eth
        )

        # ---- Likelihood ----------------------------------------------------
        sigma_y = pm.HalfNormal("sigma_y", sigma=1.0)
        pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y, dims="obs")

    meta = {
        "country_labels": country_labels,
        "region_labels": region_labels,
        "n_country": n_country,
        "n_region": n_region,
        "n_obs": len(df),
    }
    return model, meta


# ---------------------------------------------------------------------------
# Sampling, summary, and forest plot
# ---------------------------------------------------------------------------
def sample(model: pm.Model) -> az.InferenceData:
    """Draw posterior samples with NUTS at the requested precision."""
    with model:
        idata = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=0.95,
            random_seed=20260430,
            return_inferencedata=True,
            progressbar=True,
        )
    return idata


def summarize_and_save(idata: az.InferenceData) -> pd.DataFrame:
    """Build the posterior summary table (means, 94% HDI, ESS-bulk, R-hat)
    for the parameters describing the associations of interest."""
    var_names = [
        "phi",
        "beta_p",
        "beta_s",
        "beta_pop",
        "beta_urb",
        "beta_hlth",
        "beta_eth",
        "mu_a",
        "sigma_region",
        "sigma_country",
        "sigma_y",
    ]
    summary = az.summary(
        idata,
        var_names=var_names,
        kind="all",
        hdi_prob=0.94,
        round_to="none",
    )
    keep = ["mean", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]
    summary = summary[keep]

    out_csv = TABLES / "bayesian_dynamic_results.csv"
    summary.to_csv(out_csv, index=True)
    print(f"[saved] {out_csv}")
    return summary


def forest_plot(idata: az.InferenceData) -> Path:
    """Forest plot of the principal coefficients of association: phi (inertia),
    beta_p (mortality), and beta_s (stringency)."""
    axes = az.plot_forest(
        idata,
        var_names=["phi", "beta_p", "beta_s"],
        combined=True,
        hdi_prob=0.94,
        figsize=(7.5, 3.0),
    )
    fig = axes[0].figure if hasattr(axes, "__iter__") else axes.figure
    fig.suptitle(
        "Posterior associations: inertia (phi) and lagged pandemic shocks",
        fontsize=11,
    )
    fig.tight_layout()
    out_pdf = FIGS / "fig_bayesian_dynamic_forest.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_pdf}")
    return out_pdf


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_and_prepare()
    print(
        f"[panel] {len(df)} country-year observations across "
        f"{df.index.get_level_values('iso3').nunique()} countries and "
        f"{df[REGION_COL].nunique()} UN sub-regions."
    )

    model, meta = build_model(df)
    print(
        f"[model] {meta['n_obs']} obs | "
        f"{meta['n_country']} countries nested in {meta['n_region']} regions."
    )

    idata = sample(model)

    # Convergence audit on the parameters of interest.
    bad_rhat = az.rhat(idata).max().to_array().max().item()
    print(f"[diag] max R-hat across all monitored parameters = {bad_rhat:.4f}")

    summary = summarize_and_save(idata)
    print("\n[summary] Posterior of association parameters:")
    print(summary.round(4))

    forest_plot(idata)


if __name__ == "__main__":
    main()
