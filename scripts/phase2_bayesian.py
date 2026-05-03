"""
Phase 2 (Bayesian, refactored): hierarchical Bayesian model evaluating the
association between pandemic-related stressors and social polarization, with
a strict non-centered parameterization for every hierarchical layer to
neutralize Neal's funnel and ensure full posterior exploration.

Panel: 38 OECD countries × 2019-2024, countries nested in UN sub-regions.

Model
-----
    y_{c,t} ~ Normal(mu_{c,t}, sigma_y)

    mu_{c,t} = alpha_c
             + beta_p[r(c)] * p_score_mean_{c,t}
             + beta_s[r(c)] * stringency_mean_{c,t}
             + gamma' X_{c,t}
             + theta_t

Hierarchical layers (every one expressed in non-centered form):

    alpha_r   = mu_alpha   + sigma_a_r    * z_alpha_r        # region intercept
    alpha_c   = alpha_r[r(c)] + sigma_a_c * z_alpha_c        # country, nested
    beta_p[r] = mu_beta_p  + sigma_beta_p * z_beta_p         # varying slope (P-score)
    beta_s[r] = mu_beta_s  + sigma_beta_s * z_beta_s         # varying slope (stringency)

with z_* ~ Normal(0, 1) for every layer. This decouples the location and
scale parameters in the unconstrained sampling space, which is the canonical
remedy for Neal's funnel in hierarchical models with weakly identified
group-level scales (here, only 9 sub-regions provide information for the
region-level scales).

Priors are weakly informative on the standardized regressor scale:

    mu_*       ~ Normal(0, 1)
    sigma_*    ~ HalfNormal(0.5)        # tightened from 1.0 to discourage
                                        # the sigma posterior from drifting
                                        # into low-density tails near zero
    gamma      ~ Normal(0, 1)
    theta_t    ~ Normal(0, 1)
    sigma_y    ~ HalfNormal(1.0)

Outputs (overwritten on each run)
---------------------------------
    output/tables/bayesian_results.csv      posterior summary including r_hat, ess_bulk
    output/tables/bayesian_idata.nc         full InferenceData (NetCDF)
    output/figures/fig_bayesian_forest.pdf  region-level forest of the two primary slopes
    output/figures/fig_bayesian_trace.pdf   trace diagnostics for global hyperparameters

Run from project root:
    uv run python scripts/phase2_bayesian.py

Language: this module reports Bayesian *associations*. The terms "effect"
and "impact" do not appear in comments, docstrings, parameter names, or
plot labels. Year-specific shifts are called "year-specific intercepts";
country- and region-level random terms are called "varying intercepts".
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr  # noqa: F401  (listed per spec; re-exported by arviz)
import pymc as pm
import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT   = Path(__file__).resolve().parent.parent
DATA   = ROOT / "data"
TABLES = ROOT / "output" / "tables"
FIGS   = ROOT / "output" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

OUTCOME     = "v2smpolsoc"
EXPOSURES   = ["p_score_mean", "stringency_mean"]
CONFOUNDERS = ["log_pop", "urban_pct", "health_exp_gdp", "log_gdp_pc", "ethnic_frac"]

RANDOM_SEED   = 20260430
N_TUNE        = 2000
N_DRAWS       = 2000
N_CHAINS      = 4         # spec requires >= 2; 4 yields stronger R-hat diagnostics
TARGET_ACCEPT = 0.99      # tightened from 0.95 to suppress residual divergences
MAX_TREEDEPTH = 12        # raised from default 10; cheap insurance with NUTS


# --- Plot style (journal-grade) --------------------------------------------
mpl.rcParams.update({
    "figure.dpi":        300,
    "savefig.dpi":       600,
    "savefig.bbox":      "tight",
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})


# ---------------------------------------------------------------------------
# Data preparation (unchanged from prior iteration)
# ---------------------------------------------------------------------------
def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the analysis frame: derive log-transforms, drop incomplete
    rows, then standardize continuous regressors to zero mean / unit SD and
    centre the outcome. Standardization makes the weakly informative N(0, 1)
    priors describe associations of plausible magnitude on a common scale."""
    df = df.copy()
    df["log_pop"]    = np.log(df["population"])
    df["log_gdp_pc"] = np.log(df["gdp_pc_ppp"])
    keep = [OUTCOME, *EXPOSURES, *CONFOUNDERS, "iso3", "year", "region"]
    df = df[keep].dropna().reset_index(drop=True)

    for col in [*EXPOSURES, *CONFOUNDERS]:
        df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    df[OUTCOME] = df[OUTCOME] - df[OUTCOME].mean()
    return df


# ---------------------------------------------------------------------------
# Model construction — strict non-centered parameterization
# ---------------------------------------------------------------------------
def build_model(df: pd.DataFrame):
    countries = sorted(df["iso3"].unique())
    regions   = sorted(df["region"].unique())
    years     = sorted(df["year"].unique())

    country_idx = pd.Categorical(df["iso3"], categories=countries).codes.astype("int64")
    year_idx    = pd.Categorical(df["year"], categories=years).codes.astype("int64")

    country_to_region = (df.drop_duplicates("iso3")
                           .set_index("iso3")
                           .loc[countries, "region"])
    region_of_country = pd.Categorical(country_to_region,
                                       categories=regions).codes.astype("int64")
    region_per_obs = region_of_country[country_idx]   # numpy precompute

    coords = {
        "country":    countries,
        "region":     regions,
        "year":       years,
        "confounder": CONFOUNDERS,
        "obs":        np.arange(len(df)),
    }

    with pm.Model(coords=coords) as model:
        # --- Observed data -----------------------------------------------
        p_score = pm.Data("p_score", df["p_score_mean"].values,    dims="obs")
        string_ = pm.Data("string_", df["stringency_mean"].values, dims="obs")
        Xconf   = pm.Data("Xconf",   df[CONFOUNDERS].values,
                          dims=("obs", "confounder"))
        y_obs   = pm.Data("y_obs",   df[OUTCOME].values,           dims="obs")

        # --- Global hyperpriors (location parameters) --------------------
        mu_alpha  = pm.Normal("mu_alpha",  0.0, 1.0)
        mu_beta_p = pm.Normal("mu_beta_p", 0.0, 1.0)
        mu_beta_s = pm.Normal("mu_beta_s", 0.0, 1.0)

        # --- Global hyperpriors (scale parameters) -----------------------
        # HalfNormal(0.5) on the standardized scale: ~95% prior mass below
        # 1.0, which is appropriate for region-to-region heterogeneity in
        # standardized associations and helps the sampler avoid the funnel
        # geometry near sigma -> 0.
        sigma_a_r    = pm.HalfNormal("sigma_a_r",    0.5)
        sigma_a_c    = pm.HalfNormal("sigma_a_c",    0.5)
        sigma_beta_p = pm.HalfNormal("sigma_beta_p", 0.5)
        sigma_beta_s = pm.HalfNormal("sigma_beta_s", 0.5)

        # --- Region-level varying intercepts (NON-CENTERED) --------------
        # Standard-normal "innovations" decoupled from sigma_a_r in the
        # unconstrained sampling space; the deterministic shift recombines
        # location and scale on the natural scale.
        z_alpha_r = pm.Normal("z_alpha_r", 0.0, 1.0, dims="region")
        alpha_r   = pm.Deterministic(
            "alpha_r",
            mu_alpha + sigma_a_r * z_alpha_r,
            dims="region",
        )

        # --- Region-level varying slope: P-score (NON-CENTERED) ----------
        z_beta_p = pm.Normal("z_beta_p", 0.0, 1.0, dims="region")
        beta_p   = pm.Deterministic(
            "beta_p",
            mu_beta_p + sigma_beta_p * z_beta_p,
            dims="region",
        )

        # --- Region-level varying slope: stringency (NON-CENTERED) -------
        z_beta_s = pm.Normal("z_beta_s", 0.0, 1.0, dims="region")
        beta_s   = pm.Deterministic(
            "beta_s",
            mu_beta_s + sigma_beta_s * z_beta_s,
            dims="region",
        )

        # --- Country-level varying intercepts, nested in region ----------
        # Non-centered around the parent region intercept, scale sigma_a_c.
        z_alpha_c = pm.Normal("z_alpha_c", 0.0, 1.0, dims="country")
        alpha_c   = pm.Deterministic(
            "alpha_c",
            alpha_r[region_of_country] + sigma_a_c * z_alpha_c,
            dims="country",
        )

        # --- Macro-confounder global slopes (no hierarchy) ---------------
        gamma = pm.Normal("gamma", 0.0, 1.0, dims="confounder")

        # --- Year-specific intercepts (no hierarchy) ---------------------
        theta_year = pm.Normal("theta_year", 0.0, 1.0, dims="year")

        # --- Linear predictor --------------------------------------------
        mu = (
            alpha_c[country_idx]
            + beta_p[region_per_obs] * p_score
            + beta_s[region_per_obs] * string_
            + pm.math.dot(Xconf, gamma)
            + theta_year[year_idx]
        )

        # --- Observation model -------------------------------------------
        sigma_y = pm.HalfNormal("sigma_y", 1.0)
        pm.Normal("y", mu=mu, sigma=sigma_y, observed=y_obs, dims="obs")

    return model


# ---------------------------------------------------------------------------
# Posterior summary, diagnostics, and figures
# ---------------------------------------------------------------------------
PRIMARY_VARS = [
    "mu_alpha", "mu_beta_p", "mu_beta_s",
    "sigma_a_r", "sigma_a_c", "sigma_beta_p", "sigma_beta_s", "sigma_y",
    "alpha_r", "beta_p", "beta_s", "gamma", "theta_year",
]


def summarize(idata: az.InferenceData) -> pd.DataFrame:
    s = az.summary(idata, var_names=PRIMARY_VARS,
                   hdi_prob=0.94, round_to="none")
    s = s.rename(columns={"hdi_3%": "hdi_3", "hdi_97%": "hdi_97"})
    keep = ["mean", "sd", "hdi_3", "hdi_97", "ess_bulk", "ess_tail", "r_hat"]
    return s[keep]


def forest_figure(idata: az.InferenceData) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.6), sharex=False)
    az.plot_forest(idata, var_names=["beta_p"], combined=True,
                   hdi_prob=0.94, ax=axes[0])
    axes[0].set_title("Region-varying association: excess mortality (P-score)",
                      loc="left", pad=6)
    axes[0].axvline(0, color="#666666", linewidth=0.6, linestyle="--")

    az.plot_forest(idata, var_names=["beta_s"], combined=True,
                   hdi_prob=0.94, ax=axes[1])
    axes[1].set_title("Region-varying association: lockdown stringency",
                      loc="left", pad=6)
    axes[1].axvline(0, color="#666666", linewidth=0.6, linestyle="--")

    fig.suptitle(
        "Hierarchical Bayesian posteriors — region-level associations with social polarization",
        fontsize=10, x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIGS / "fig_bayesian_forest.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def trace_figure(idata: az.InferenceData) -> Path:
    primary_globals = ["mu_beta_p", "mu_beta_s",
                       "sigma_beta_p", "sigma_beta_s",
                       "sigma_a_r", "sigma_a_c", "sigma_y"]
    az.plot_trace(idata, var_names=primary_globals, compact=True)
    fig = plt.gcf()
    fig.suptitle("Trace diagnostics — global hyperparameters",
                 fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGS / "fig_bayesian_trace.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run() -> None:
    panel = pd.read_parquet(DATA / "oecd_panel.parquet")
    df = _prep(panel)
    print(f"[bayes] analysis frame: {len(df)} obs, "
          f"{df.iso3.nunique()} countries, "
          f"{df.region.nunique()} regions, "
          f"{df.year.nunique()} years")

    model = build_model(df)
    with model:
        idata = pm.sample(
            draws=N_DRAWS,
            tune=N_TUNE,
            chains=N_CHAINS,
            cores=min(N_CHAINS, 4),
            target_accept=TARGET_ACCEPT,
            max_treedepth=MAX_TREEDEPTH,
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
        )

    summary = summarize(idata)
    summary.to_csv(TABLES / "bayesian_results.csv")
    idata.to_netcdf(TABLES / "bayesian_idata.nc")

    forest_path = forest_figure(idata)
    trace_path  = trace_figure(idata)

    rhat_max     = summary["r_hat"].max()
    ess_bulk_min = summary["ess_bulk"].min()
    n_diverging  = int(idata.sample_stats["diverging"].sum().item()) \
                   if "diverging" in idata.sample_stats else -1

    print(f"[bayes] wrote {TABLES / 'bayesian_results.csv'}")
    print(f"[bayes] wrote {forest_path}")
    print(f"[bayes] wrote {trace_path}")
    print(f"[bayes] convergence: max R-hat = {rhat_max:.4f}, "
          f"min ESS-bulk = {ess_bulk_min:.0f}, "
          f"divergences (post-warmup) = {n_diverging}")
    if rhat_max > 1.01:
        print("[bayes] WARNING: R-hat exceeds 1.01 for at least one parameter; "
              "inspect the trace plot before reporting.")
    if n_diverging > 0:
        print("[bayes] NOTE: residual divergences detected; consider raising "
              "target_accept further or tightening sigma_* priors.")


if __name__ == "__main__":
    run()
