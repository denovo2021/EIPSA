# EIPSA — Hierarchical Bayesian Dynamic Panel Analysis

Reproducibility package for the manuscript estimating the association
between pre-pandemic social cohesion (V-Dem `v2smpolsoc`, 2019) and
COVID-19 lockdown stringency (OxCGRT Stringency Index) across OECD
countries between 2019 and 2024.

Inference is performed with a Hierarchical Bayesian Dynamic Panel Model
implemented in PyMC v5. Country (`iso3`) units are partially pooled
within UN sub-regions; a within-country autoregressive term absorbs
macro-structural inertia. All hierarchical layers are parameterised in
non-centred form. Sampling uses NUTS with 4 chains, 2000 tuning steps,
2000 draws, and `target_accept = 0.95`. Convergence is audited at
`max R-hat <= 1.01` and `min bulk ESS > 400`. All posterior intervals
are reported at the strict 95% HDI.

## Repository structure

```
eipsa/
├── README.md
├── pyproject.toml          # project metadata + pinned dependencies
├── requirements.txt        # mirror of dependencies for plain pip
├── .gitignore
├── main.py                 # end-to-end pipeline (fit -> tables -> figures)
├── data/
│   └── oecd_panel.csv      # analytic OECD country-year panel (38 countries, 2019-2024)
└── scripts/
    ├── __init__.py
    ├── data_prep.py        # panel loader and lagged-regressor construction
    ├── bayesian_model.py   # PyMC v5 model (Lag-1 primary, Lag-2 sensitivity)
    └── plot_results.py     # 95% HDI tables, Figure 1, Figure 2
```

Runtime artefacts written by `main.py` are placed under `output/` and
are excluded from version control.

## Data sources

The analytic file `data/oecd_panel.csv` is a country-year panel with the
following columns. The two pandemic-era exposures are constructed by
this project; every other variable is sourced verbatim from the
indicated public dataset.

| Variable                                | Source                                                |
| --------------------------------------- | ----------------------------------------------------- |
| `v2smpolsoc` (outcome)                  | V-Dem v15 (`v2smpolsoc` social cohesion indicator)    |
| `stringency_mean`, `stringency_max`     | OxCGRT compact national data, annualised             |
| `p_score_mean`, `p_score_peak`          | OWID excess mortality (P-score, annualised)           |
| `cum_excess_per_100k`                   | OWID excess mortality (cumulative, per 100k)          |
| `population`, `urban_pct`, `gdp_pc_ppp` | World Bank WDI                                        |
| `health_exp_gdp`, `pop_density`         | World Bank WDI                                        |
| `ethnic_frac`                           | Historical Index of Ethnic Fractionalisation (HIEF)   |
| `region`                                | UN sub-region classification                          |

## Environment setup

The pipeline requires Python 3.12 or newer.

### With `uv` (recommended)

```bash
uv sync
uv run python main.py
```

### With `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Reproducing the manuscript outputs

A single command reproduces every numerical result:

```bash
python main.py
```

The pipeline first fits the primary (Lag-1) specification and then the
Lag-2 sensitivity specification, persists each trace as NetCDF under
`output/`, and writes:

| Output                  | Path                                                          |
| ----------------------- | ------------------------------------------------------------- |
| Table 1                 | `output/tables/posterior_summary_main_95hdi.csv`              |
| Figure 1                | `output/figures/fig1_forest_main_95hdi.pdf`                   |
| Figure 2                | `output/figures/fig2_selection_effect_scatter.pdf`            |
| Supplementary Table S1  | `output/tables/posterior_summary_si_lag2_95hdi.csv`           |
| Supplementary Table S2  | `output/tables/convergence_diagnostics.csv`                   |
| Pearson statistic (Fig 2)| `output/tables/selection_effect_pearson_n37.csv`              |
| Lag-1 trace             | `output/idata_main_lag1.nc`                                   |
| Lag-2 trace             | `output/idata_sensitivity_lag2.nc`                            |

Subsequent runs re-use the cached NetCDF traces and skip sampling. To
force a fresh fit, delete the `.nc` files under `output/` before
re-running `main.py`. The random seed (`SEED = 20260503` in
`scripts/bayesian_model.py`) is fixed for reproducibility.

## Notes on the analytic sample

The Bayesian models are estimated on the complete-case panel
(`N_obs = 152` for the Lag-1 specification across 37 countries and
their UN sub-regions). Türkiye (`TUR`) is excluded because the lagged
log COVID-mortality regressor required by the dynamic model is not
reported for Türkiye over 2019–2024. The same exclusion is applied to
the Figure 2 cross-section so the bivariate association in Figure 2 is
estimated on the identical sample as the model-based association in
Figure 1.

## Terminology

Throughout the code, comments, and this README, the temporal and
cross-country signal estimated by the model is referred to as an
*association*. The Pearson coefficient annotated on Figure 2 is
reported using the conventional symbol *r* and quantifies the strength
of the bivariate association on the analytic sample.
