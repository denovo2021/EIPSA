"""Data preparation for the EIPSA Hierarchical Bayesian Dynamic Panel Model.

Loads the analytic OECD country-year panel and constructs the within-country
lagged regressors required by the Lag-1 (primary) and Lag-2 (sensitivity)
specifications. The lagged terms encode the temporal association between
pandemic-era exposures and subsequent social cohesion, conditional on
country- and region-level partial pooling.

The canonical input is ``data/oecd_panel.csv``, a country-year panel
covering 38 OECD economies between 2019 and 2024.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

OUTCOME = "v2smpolsoc"
COVARIATES = ["log_pop", "urban_pct", "health_exp_gdp", "ethnic_frac"]
REGION_COL = "region"
PANEL_PATH = DATA / "oecd_panel.csv"


def load_panel(path: Path = PANEL_PATH) -> pd.DataFrame:
    """Load the raw OECD country-year panel from disk.

    Returns the panel sorted by (iso3, year) with the log-population
    covariate derived on the fly if it is not already present.
    """
    df = pd.read_csv(path)
    if "log_pop" not in df.columns:
        df["log_pop"] = np.log(df["population"])
    return df.sort_values(["iso3", "year"]).reset_index(drop=True)


def _make_lagged_frame(df: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Construct within-country lagged regressors at the requested horizon.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_panel`.
    lag : int
        Number of years to shift the outcome and pandemic-exposure series.
        ``lag=1`` yields the primary specification; ``lag=2`` yields the
        Lag-2 sensitivity specification.
    """
    indexed = df.set_index(["iso3", "year"])
    indexed[f"{OUTCOME}_lag{lag}"] = indexed.groupby(level=0)[OUTCOME].shift(lag)
    indexed[f"p_score_mean_lag{lag}"] = (
        indexed.groupby(level=0)["p_score_mean"].shift(lag)
    )
    indexed[f"stringency_mean_lag{lag}"] = (
        indexed.groupby(level=0)["stringency_mean"].shift(lag)
    )
    needed = [
        OUTCOME,
        f"{OUTCOME}_lag{lag}",
        f"p_score_mean_lag{lag}",
        f"stringency_mean_lag{lag}",
        *COVARIATES,
        REGION_COL,
    ]
    return indexed.dropna(subset=needed).copy()


def prepare_lag1(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the analytic frame for the primary (Lag-1) model."""
    if df is None:
        df = load_panel()
    return _make_lagged_frame(df, lag=1)


def prepare_lag2(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the analytic frame for the Lag-2 sensitivity model."""
    if df is None:
        df = load_panel()
    return _make_lagged_frame(df, lag=2)


def zscore(s: pd.Series) -> np.ndarray:
    """Population z-score (ddof=0) used to standardize every regressor.

    Standardization keeps all coefficients of association on a common
    scale and lets the Normal(0, 1) priors on the slope parameters
    operate as weakly informative shrinkage.
    """
    return ((s - s.mean()) / s.std(ddof=0)).to_numpy()
