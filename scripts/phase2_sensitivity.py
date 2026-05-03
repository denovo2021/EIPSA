"""
Phase 2 (Bayesian sensitivity analyses): two robustness checks for the
hierarchical dynamic panel model evaluating the association between
pandemic-related stressors and social polarization (v2smpolsoc).

    Model 1 -- Lag-2 dynamic panel model
        Replaces 1-year lags with 2-year lags for the outcome and the
        primary exposures, probing whether the inertia / acute-shock
        associations identified at lag-1 persist at a longer horizon.

    Model 2 -- Bayesian effect-modification model
        Adds product terms between the lag-1 exposures and pre-pandemic
        baseline cohesion (cohesion_2019) to assess whether the lagged
        pandemic-shock associations are modified by baseline cohesion.

Both models share the hierarchical backbone of the primary specification:
countries nested in UN sub-regions, with strict NON-CENTERED parameterization
of every hierarchical layer (Neal's-funnel safe).

Run from the project root:
    python scripts/phase2_sensitivity.py
"""
from __future__ import annotations
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TABLES = ROOT / "output" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

OUTCOME = "v2smpolsoc"
COVARIATES = ["log_pop", "urban_pct", "health_exp_gdp", "ethnic_frac"]
REGION_COL = "region"  # UN sub-region label

SAMPLE_KW = dict(
    draws=2000,
    tune=2000,
    chains=4,
    target_accept=0.95,
    random_seed=20260430,
    return_inferencedata=True,
    progressbar=True,
)


# ---------------------------------------------------------------------------
# Data loading and feature construction
# ---------------------------------------------------------------------------
def load_panel() -> pd.DataFrame:
    """Load the OECD panel, ensure log_pop exists, and index by (iso3, year)."""
    df = pd.read_parquet(DATA / "oecd_panel.parquet")
    if "log_pop" not in df.columns:
        df["log_pop"] = np.log(df["population"])
    return df.sort_values(["iso3", "year"]).set_index(["iso3", "year"])


def build_lag2_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Construct 2-year within-country lags of the outcome and exposures and
    restrict to complete cases on the variables used in Model 1."""
    out = df.copy()
    out["v2smpolsoc_lag2"] = out.groupby(level=0)[OUTCOME].shift(2)
    out["p_score_mean_lag2"] = out.groupby(level=0)["p_score_mean"].shift(2)
    out["stringency_mean_lag2"] = out.groupby(level=0)["stringency_mean"].shift(2)
    needed = [
        OUTCOME,
        "v2smpolsoc_lag2",
        "p_score_mean_lag2",
        "stringency_mean_lag2",
        *COVARIATES,
        REGION_COL,
    ]
    return out.dropna(subset=needed).copy()


def build_interaction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Construct 1-year within-country lags of the outcome and exposures,
    broadcast pre-pandemic cohesion (v2smpolsoc at year==2019) to every
    country-year, and form the product terms used in Model 2."""
    out = df.copy()
    out["v2smpolsoc_lag1"] = out.groupby(level=0)[OUTCOME].shift(1)
    out["p_score_mean_lag1"] = out.groupby(level=0)["p_score_mean"].shift(1)
    out["stringency_mean_lag1"] = out.groupby(level=0)["stringency_mean"].shift(1)

    cohesion_2019 = (
        out[OUTCOME]
        .reset_index()
        .query("year == 2019")
        .set_index("iso3")[OUTCOME]
        .rename("cohesion_2019")
    )
    out = out.join(cohesion_2019, on="iso3")

    out["inter_p"] = out["p_score_mean_lag1"] * out["cohesion_2019"]
    out["inter_s"] = out["stringency_mean_lag1"] * out["cohesion_2019"]

    needed = [
        OUTCOME,
        "v2smpolsoc_lag1",
        "p_score_mean_lag1",
        "stringency_mean_lag1",
        "inter_p",
        "inter_s",
        "cohesion_2019",
        *COVARIATES,
        REGION_COL,
    ]
    return out.dropna(subset=needed).copy()


def zstd(s: pd.Series) -> np.ndarray:
    """Z-standardize a series (mean 0, sd 1) for NUTS sampler stability."""
    return ((s - s.mean()) / s.std(ddof=0)).to_numpy()


# ---------------------------------------------------------------------------
# Hierarchical backbone shared by both sensitivity models
# ---------------------------------------------------------------------------
def make_hierarchy(df: pd.DataFrame):
    """Return category-coded indices for the country and region hierarchy."""
    countries = df.index.get_level_values("iso3").astype("category")
    country_codes = countries.codes
    country_labels = countries.categories.to_numpy()

    country_to_region = (
        df.reset_index()[["iso3", REGION_COL]]
        .drop_duplicates("iso3")
        .set_index("iso3")
        .loc[country_labels, REGION_COL]
        .astype("category")
    )
    region_labels = country_to_region.cat.categories.to_numpy()
    c2r_idx = country_to_region.cat.codes.to_numpy()

    coords = {
        "country": country_labels,
        "region": region_labels,
        "obs": np.arange(len(df)),
    }
    return coords, country_codes, c2r_idx


def add_hierarchical_intercepts(c2r_idx: np.ndarray) -> pm.Distribution:
    """Add the non-centered region-then-country hierarchical intercepts to the
    currently active PyMC model context and return a_country."""
    mu_a = pm.Normal("mu_a", mu=0.0, sigma=2.0)

    # Region-level intercepts (non-centered).
    sigma_region = pm.HalfNormal("sigma_region", sigma=1.0)
    z_region = pm.Normal("z_region", mu=0.0, sigma=1.0, dims="region")
    a_region = pm.Deterministic(
        "a_region", mu_a + z_region * sigma_region, dims="region"
    )

    # Country-level intercepts nested in region (non-centered).
    sigma_country = pm.HalfNormal("sigma_country", sigma=1.0)
    z_country = pm.Normal("z_country", mu=0.0, sigma=1.0, dims="country")
    a_country = pm.Deterministic(
        "a_country",
        a_region[c2r_idx] + z_country * sigma_country,
        dims="country",
    )
    return a_country


# ---------------------------------------------------------------------------
# Model 1 -- Lag-2 dynamic panel
# ---------------------------------------------------------------------------
def fit_lag2_model(df: pd.DataFrame) -> az.InferenceData:
    coords, country_idx, c2r_idx = make_hierarchy(df)

    y = df[OUTCOME].to_numpy()
    x_lag2 = zstd(df["v2smpolsoc_lag2"])
    x_p2 = zstd(df["p_score_mean_lag2"])
    x_s2 = zstd(df["stringency_mean_lag2"])
    x_pop = zstd(df["log_pop"])
    x_urb = zstd(df["urban_pct"])
    x_hlth = zstd(df["health_exp_gdp"])
    x_eth = zstd(df["ethnic_frac"])

    with pm.Model(coords=coords) as model:
        a_country = add_hierarchical_intercepts(c2r_idx)

        # Coefficients of association (lag-2 horizon).
        phi = pm.Normal("phi", mu=0.0, sigma=1.0)
        beta_p2 = pm.Normal("beta_p2", mu=0.0, sigma=1.0)
        beta_s2 = pm.Normal("beta_s2", mu=0.0, sigma=1.0)
        beta_pop = pm.Normal("beta_pop", mu=0.0, sigma=1.0)
        beta_urb = pm.Normal("beta_urb", mu=0.0, sigma=1.0)
        beta_hlth = pm.Normal("beta_hlth", mu=0.0, sigma=1.0)
        beta_eth = pm.Normal("beta_eth", mu=0.0, sigma=1.0)

        mu = (
            a_country[country_idx]
            + phi * x_lag2
            + beta_p2 * x_p2
            + beta_s2 * x_s2
            + beta_pop * x_pop
            + beta_urb * x_urb
            + beta_hlth * x_hlth
            + beta_eth * x_eth
        )
        sigma_y = pm.HalfNormal("sigma_y", sigma=1.0)
        pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y, dims="obs")

        idata = pm.sample(**SAMPLE_KW)

    return idata


# ---------------------------------------------------------------------------
# Model 2 -- Bayesian effect-modification (interaction with cohesion_2019)
# ---------------------------------------------------------------------------
def fit_interaction_model(df: pd.DataFrame) -> az.InferenceData:
    coords, country_idx, c2r_idx = make_hierarchy(df)

    y = df[OUTCOME].to_numpy()
    x_lag1 = zstd(df["v2smpolsoc_lag1"])
    x_p = zstd(df["p_score_mean_lag1"])
    x_s = zstd(df["stringency_mean_lag1"])
    # The product terms are standardized as a single column (after construction)
    # rather than as the product of standardized inputs, so the coefficient is
    # interpretable on the scale of the constructed interaction.
    x_ip = zstd(df["inter_p"])
    x_is = zstd(df["inter_s"])
    x_pop = zstd(df["log_pop"])
    x_urb = zstd(df["urban_pct"])
    x_hlth = zstd(df["health_exp_gdp"])
    x_eth = zstd(df["ethnic_frac"])

    with pm.Model(coords=coords) as model:
        # The main term for cohesion_2019 is time-invariant within country and
        # is absorbed by a_country; only the product terms are separately
        # identified alongside the time-varying main exposures.
        a_country = add_hierarchical_intercepts(c2r_idx)

        phi = pm.Normal("phi", mu=0.0, sigma=1.0)
        beta_p = pm.Normal("beta_p", mu=0.0, sigma=1.0)
        beta_s = pm.Normal("beta_s", mu=0.0, sigma=1.0)
        beta_inter_p = pm.Normal("beta_inter_p", mu=0.0, sigma=1.0)
        beta_inter_s = pm.Normal("beta_inter_s", mu=0.0, sigma=1.0)
        beta_pop = pm.Normal("beta_pop", mu=0.0, sigma=1.0)
        beta_urb = pm.Normal("beta_urb", mu=0.0, sigma=1.0)
        beta_hlth = pm.Normal("beta_hlth", mu=0.0, sigma=1.0)
        beta_eth = pm.Normal("beta_eth", mu=0.0, sigma=1.0)

        mu = (
            a_country[country_idx]
            + phi * x_lag1
            + beta_p * x_p
            + beta_s * x_s
            + beta_inter_p * x_ip
            + beta_inter_s * x_is
            + beta_pop * x_pop
            + beta_urb * x_urb
            + beta_hlth * x_hlth
            + beta_eth * x_eth
        )
        sigma_y = pm.HalfNormal("sigma_y", sigma=1.0)
        pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y, dims="obs")

        idata = pm.sample(**SAMPLE_KW)

    return idata


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
KEEP_COLS = ["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]


def save_summary(idata: az.InferenceData, var_names: list[str], path: Path) -> pd.DataFrame:
    """Build the posterior summary table and write it to disk."""
    summary = az.summary(idata, var_names=var_names, kind="all", hdi_prob=0.94, round_to="none")
    summary = summary[KEEP_COLS]
    summary.to_csv(path, index=True)
    print(f"[saved] {path}")
    return summary


def banner(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    panel = load_panel()

    # -------- Model 1: Lag-2 dynamic panel ---------------------------------
    banner("Model 1 -- Lag-2 dynamic panel (sensitivity to longer horizon)")
    df_lag2 = build_lag2_frame(panel)
    print(
        f"[panel] lag-2 frame: {len(df_lag2)} obs across "
        f"{df_lag2.index.get_level_values('iso3').nunique()} countries."
    )
    idata_lag2 = fit_lag2_model(df_lag2)
    summary_lag2 = save_summary(
        idata_lag2,
        var_names=[
            "phi", "beta_p2", "beta_s2",
            "beta_pop", "beta_urb", "beta_hlth", "beta_eth",
            "mu_a", "sigma_region", "sigma_country", "sigma_y",
        ],
        path=TABLES / "sensitivity_lag2_results.csv",
    )
    print("\n[Model 1] Posterior summary of association parameters:")
    print(summary_lag2.loc[["phi", "beta_p2", "beta_s2"]].round(4))

    # -------- Model 2: Effect-modification (interaction) -------------------
    banner("Model 2 -- Effect modification by pre-pandemic cohesion (cohesion_2019)")
    df_inter = build_interaction_frame(panel)
    print(
        f"[panel] interaction frame: {len(df_inter)} obs across "
        f"{df_inter.index.get_level_values('iso3').nunique()} countries."
    )
    idata_inter = fit_interaction_model(df_inter)
    summary_inter = save_summary(
        idata_inter,
        var_names=[
            "phi", "beta_p", "beta_s", "beta_inter_p", "beta_inter_s",
            "beta_pop", "beta_urb", "beta_hlth", "beta_eth",
            "mu_a", "sigma_region", "sigma_country", "sigma_y",
        ],
        path=TABLES / "sensitivity_interaction_results.csv",
    )
    print("\n[Model 2] Posterior summary of association parameters:")
    print(summary_inter.loc[["phi", "beta_p", "beta_s", "beta_inter_p", "beta_inter_s"]].round(4))


if __name__ == "__main__":
    main()
