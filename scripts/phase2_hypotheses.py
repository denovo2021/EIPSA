"""
Phase 2 (exploratory): rapid panel tests of three alternative hypotheses for the
association between pandemic shocks and social polarization (v2smpolsoc).

Models fit with linearmodels.PanelOLS using cluster-robust (by entity) standard
errors. EntityEffects / TimeEffects are toggled per-model as documented below.

Run from project root:
    python scripts/phase2_hypotheses.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

OUTCOME = "v2smpolsoc"


def load_panel() -> pd.DataFrame:
    """Load the OECD panel and set a (iso3, year) MultiIndex."""
    df = pd.read_parquet(DATA / "oecd_panel.parquet")
    # Construct covariates that are functions of raw measures (log transforms).
    df["log_pop"] = np.log(df["population"])
    df["log_gdp_pc"] = np.log(df["gdp_pc_ppp"])
    df = df.sort_values(["iso3", "year"]).set_index(["iso3", "year"])
    return df


def build_analytic_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lagged exposures, lagged outcome, the economic-shock measure,
    and a time-invariant pre-pandemic cohesion covariate."""
    out = df.copy()

    # 1-year lag of the outcome (for the dynamic autoregressive specification).
    out["v2smpolsoc_lag1"] = out.groupby(level=0)[OUTCOME].shift(1)

    # 1-year lags of the primary exposures (delayed-association specification).
    out["stringency_mean_lag1"] = out.groupby(level=0)["stringency_mean"].shift(1)
    out["p_score_mean_lag1"] = out.groupby(level=0)["p_score_mean"].shift(1)

    # First difference of log GDP per capita -> economic shock (annual growth).
    out["gdp_growth"] = out.groupby(level=0)["log_gdp_pc"].diff()

    # Pre-pandemic social cohesion (2019 baseline), broadcast to all country-years.
    # Used as a time-invariant moderator for the effect-modification hypothesis.
    cohesion_2019 = (
        out[OUTCOME]
        .reset_index()
        .query("year == 2019")
        .set_index("iso3")[OUTCOME]
        .rename("cohesion_2019")
    )
    out = out.join(cohesion_2019, on="iso3")

    return out


def fit_h1_dynamic(df: pd.DataFrame):
    """H1 -- dynamic autoregressive specification.

    The lagged outcome is included to characterise persistence in social
    polarization. EntityEffects are deliberately omitted because the within
    transformation combined with a lagged dependent variable induces Nickell
    bias in short panels; we instead rely on observed time-varying covariates
    plus TimeEffects to absorb common-shock confounders.
    """
    cols = [
        OUTCOME,
        "v2smpolsoc_lag1",
        "p_score_mean_lag1",
        "stringency_mean_lag1",
        "log_pop",
        "urban_pct",
        "health_exp_gdp",
        "ethnic_frac",
    ]
    d = df[cols].dropna()
    y = d[OUTCOME]
    X = d.drop(columns=[OUTCOME])
    X = X.assign(const=1.0)
    mod = PanelOLS(y, X, entity_effects=False, time_effects=True, drop_absorbed=True)
    return mod.fit(cov_type="clustered", cluster_entity=True)


def fit_h2_modification(df: pd.DataFrame):
    """H2 -- effect modification of pandemic-shock association by pre-pandemic
    cohesion (buffering hypothesis).

    Product terms with cohesion_2019 are included to assess whether the
    association of the exposures with v2smpolsoc varies across strata of
    baseline cohesion. The main term for cohesion_2019 is time-invariant and
    is therefore absorbed by EntityEffects; the product terms remain identified
    because the exposures vary within country.
    """
    cols = [
        OUTCOME,
        "p_score_mean",
        "stringency_mean",
        "cohesion_2019",
        "log_pop",
        "urban_pct",
        "health_exp_gdp",
        "log_gdp_pc",
        "ethnic_frac",
    ]
    d = df[cols].dropna().copy()
    # Construct product (interaction) terms for the moderator analysis.
    d["stringency_x_cohesion19"] = d["stringency_mean"] * d["cohesion_2019"]
    d["pscore_x_cohesion19"] = d["p_score_mean"] * d["cohesion_2019"]

    y = d[OUTCOME]
    X = d[
        [
            "p_score_mean",
            "stringency_mean",
            "stringency_x_cohesion19",
            "pscore_x_cohesion19",
            "log_pop",
            "urban_pct",
            "health_exp_gdp",
            "log_gdp_pc",
            "ethnic_frac",
        ]
    ].assign(const=1.0)
    mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
    return mod.fit(cov_type="clustered", cluster_entity=True)


def fit_h3_economic(df: pd.DataFrame):
    """H3 -- economic shock as alternative exposure.

    Annual GDP-per-capita growth (first-differenced log GDP) is added alongside
    the pandemic-shock exposures to assess whether the contemporaneous
    association of pandemic measures with polarization persists after
    adjustment for the concurrent macroeconomic shock.
    """
    cols = [
        OUTCOME,
        "gdp_growth",
        "p_score_mean",
        "stringency_mean",
        "log_pop",
        "urban_pct",
        "health_exp_gdp",
        "ethnic_frac",
    ]
    d = df[cols].dropna()
    y = d[OUTCOME]
    X = d.drop(columns=[OUTCOME]).assign(const=1.0)
    mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
    return mod.fit(cov_type="clustered", cluster_entity=True)


def _banner(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


def main() -> None:
    panel = load_panel()
    panel = build_analytic_frame(panel)

    _banner("Model 1 (H1) -- Dynamic autoregressive association; TimeEffects only")
    res1 = fit_h1_dynamic(panel)
    print(res1.summary)

    _banner("Model 2 (H2) -- Effect modification by pre-pandemic cohesion")
    res2 = fit_h2_modification(panel)
    print(res2.summary)

    _banner("Model 3 (H3) -- Economic shock as alternative exposure")
    res3 = fit_h3_economic(panel)
    print(res3.summary)


if __name__ == "__main__":
    main()
