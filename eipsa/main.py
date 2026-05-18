"""End-to-end pipeline for the EIPSA Hierarchical Bayesian Dynamic Panel
Model.

Running this file from the repository root reproduces every numerical
result, table, and figure that appears in the manuscript and the
supplementary information:

    Table 1                  output/tables/posterior_summary_main_95hdi.csv
    Figure 1                 output/figures/fig1_forest_main_95hdi.pdf
    Figure 2                 output/figures/fig2_selection_effect_scatter.pdf
    Supplementary Table S1   output/tables/posterior_summary_si_lag2_95hdi.csv
    Supplementary Table S2   output/tables/convergence_diagnostics.csv

Two NetCDF traces are also persisted so the figures and tables can be
regenerated without re-running the sampler:

    output/idata_main_lag1.nc
    output/idata_sensitivity_lag2.nc

Usage
-----
    uv run python main.py
or
    python main.py
"""
from __future__ import annotations

from pathlib import Path

import arviz as az

from scripts import bayesian_model, data_prep, plot_results

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"

IDATA_LAG1 = OUTPUT / "idata_main_lag1.nc"
IDATA_LAG2 = OUTPUT / "idata_sensitivity_lag2.nc"


def _fit_or_load(lag: int, idata_path: Path) -> az.InferenceData:
    """Sample the requested specification, or reload it from disk if cached."""
    if idata_path.exists():
        print(f"[load] {idata_path}")
        return az.from_netcdf(idata_path)

    prepare = data_prep.prepare_lag1 if lag == 1 else data_prep.prepare_lag2
    df = prepare()
    n_obs = len(df)
    n_countries = df.index.get_level_values("iso3").nunique()
    n_regions = df[data_prep.REGION_COL].nunique()
    print(
        f"[panel lag-{lag}] N={n_obs} obs, "
        f"{n_countries} countries, {n_regions} UN sub-regions."
    )

    idata = bayesian_model.fit(df, lag=lag)
    bayesian_model.save_idata(idata, idata_path)
    return idata


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # --- 1. Fit (or reload) the two Bayesian specifications ----------------
    idata_lag1 = _fit_or_load(lag=1, idata_path=IDATA_LAG1)
    idata_lag2 = _fit_or_load(lag=2, idata_path=IDATA_LAG2)

    # --- 2. Tables ---------------------------------------------------------
    plot_results.export_posterior_summary(
        idata_lag1, TABLES / "posterior_summary_main_95hdi.csv"
    )
    plot_results.export_posterior_summary(
        idata_lag2, TABLES / "posterior_summary_si_lag2_95hdi.csv"
    )
    plot_results.export_convergence_diagnostics(
        {"lag1_main": idata_lag1, "lag2_sensitivity": idata_lag2},
        TABLES / "convergence_diagnostics.csv",
    )

    # --- 3. Figures --------------------------------------------------------
    plot_results.figure1_forest(
        idata_lag1, FIGURES / "fig1_forest_main_95hdi.pdf"
    )
    plot_results.figure2_selection_effect(
        FIGURES / "fig2_selection_effect_scatter.pdf",
        out_stats_csv=TABLES / "selection_effect_pearson_n37.csv",
    )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
