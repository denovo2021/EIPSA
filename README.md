# EIPSA — Pre-pandemic Social Cohesion and COVID-19 Lockdown Stringency

Replication materials for the manuscript

> **Kawahara T, Fujiwara T.** *Pre-pandemic social cohesion was associated with COVID-19 lockdown stringency, but pandemic shocks did not shift polarization trajectories: a Bayesian dynamic panel analysis of 37 OECD countries.*
> Revised manuscript under peer review (2026). The journal reference and DOI will be added here upon publication.

This repository contains all code and the analysis-ready data required to reproduce the Hierarchical Bayesian Dynamic Panel Model, the five sensitivity analyses (two pre-specified; three added in revision), the interpretation quantities, and the figures and tables of the manuscript and Supplementary Information.

---

## Study summary

We estimate a Hierarchical Bayesian Dynamic Panel Model on a country-year panel of the 38 OECD member states for 2019–2024, integrating:

- **Outcome:** V-Dem *polarization of society* (`v2smpolsoc`, v15; expert-coded, higher = weaker societal polarization), used throughout as the measure-anchored definition of "social cohesion";
- **Exposure 1 — pandemic mortality burden:** the country-year mean of the monthly all-age **excess-mortality P-score** (percentage deviation of all-cause deaths from the projected pre-pandemic baseline), compiled by Our World in Data from the Human Mortality Database and the World Mortality Dataset (Karlinsky & Kobak, 2021);
- **Exposure 2 — policy stringency:** the country-year mean OxCGRT Stringency Index (0–100);
- Four macro-covariates (log population, urban %, health expenditure % GDP, ethnic fractionalization), selected a priori as plausible common causes of exposures and outcome.

Both exposures enter lagged by one year. The model accommodates **macro-structural inertia** (autoregressive term on lagged cohesion) and **spatial non-independence** (UN sub-regional partial pooling under a strictly non-centred two-level hierarchy).

**Estimation sample (exact accounting).** 38 countries × 6 years = 228 source country-years → **111 estimation country-years (37 countries × 2021–2023)**. Excluded by construction: all 2019 rows (initial lagged outcome), all 2020 rows (lagged exposures refer to 2019, before either exposure existed), all 2024 rows (OxCGRT discontinued after December 2022), and Türkiye entirely (no OWID excess-mortality series). Lag-2 sample: 96; joint distributed-lag sample: 74. Derivation: `scripts/interpretation_and_accounting.py`.

**Main findings.** Year-on-year persistence dominates the trajectory of societal polarization (φ = 1.29 per SD of the lagged outcome; native-scale persistence 0.93, 95% HDI 0.87–1.00; half-life of deviations ≈ 10 years), while neither one-year-lagged excess mortality (β = 0.05, 95% HDI −0.04 to 0.13) nor lockdown stringency (β = 0.00, −0.06 to 0.07) is credibly associated with the residual, non-inertial component — informative nulls bounded below ~0.1 SD per SD of exposure. Descriptively, pre-pandemic (2019) cohesion was strongly negatively associated with mean 2020–2021 lockdown stringency (r = −0.51, p = 0.001), a selection pattern we interpret non-causally.

---

## Repository structure

```
EIPSA/
├── data/
│   ├── oecd_panel.parquet                        # Analysis-ready panel (primary pipeline)
│   ├── oecd_panel.csv                            # CSV mirror
│   ├── EIPSA_OECD_panel_2019_2024.csv            # Extended panel (incl. v2cacamps)
│   └── raw/                                      # Source extracts (excess mortality, OxCGRT, HIEF)
├── scripts/
│   ├── fit_main_model_correct.py                 # PRIMARY model (Table 1, Figure 1)
│   ├── fit_sensitivity_lag2.py                   # SA1: Lag-2 horizon (SI Appendix B, Table S1)
│   ├── fit_sensitivity_interaction.py            # SA2: Effect modification (SI Appendix C, Table S2)
│   ├── fit_sensitivity_measurement_error.py      # SA3: V-Dem coder-uncertainty propagation
│   │                                             #      (SI Appendix E, Table S3) [added in revision]
│   ├── fit_sensitivity_alt_outcome.py            # SA4: Alternative outcome v2cacamps, reverse-scored
│   │                                             #      (SI Appendix F, Table S4) [added in revision]
│   ├── fit_sensitivity_distributed_lag.py        # SA5: Joint lag-1 + lag-2 terms
│   │                                             #      (SI Appendix G, Table S5) [added in revision]
│   ├── interpretation_and_accounting.py          # Native-scale persistence & half-life, SD-unit
│   │                                             #      effect bounds, residual lag-1 autocorrelation
│   │                                             #      (SI Appendix G), exact sample accounting
│   ├── 02_selection_effect.py                    # Figure 2 (descriptive selection pattern)
│   └── final_outputs.py                          # 95% HDI summary tables + figure regeneration
└── output/                                       # Created at run time (not tracked)
```

## Requirements & installation

Python ≥ 3.12 with **PyMC v5** (NUTS), ArviZ, pandas, NumPy, pyarrow (and matplotlib/seaborn for figures).

```bash
git clone https://github.com/denovo2021/EIPSA.git
cd EIPSA
pip install -r requirements.txt
```

## Reproducing the analysis

```bash
# 1. Primary Lag-1 model  ->  output/idata_main_correct.nc, Table 1
python scripts/fit_main_model_correct.py

# 2. Pre-specified sensitivity analyses (SI Appendices B, C)
python scripts/fit_sensitivity_lag2.py
python scripts/fit_sensitivity_interaction.py

# 3. Revision-added sensitivity analyses (SI Appendices E, F, G)
python scripts/fit_sensitivity_measurement_error.py
python scripts/fit_sensitivity_alt_outcome.py
python scripts/fit_sensitivity_distributed_lag.py

# 4. Interpretation quantities, residual-autocorrelation diagnostic,
#    and exact estimation-sample accounting (requires step 1)
python scripts/interpretation_and_accounting.py

# 5. Descriptive selection pattern (Figure 2)
python scripts/02_selection_effect.py
```

**Sampler settings (all Bayesian models).** NUTS, 4 chains × 2,000 tuning + 2,000 post-warmup draws (the measurement-error model uses 3,000 tuning), target acceptance 0.95 (0.99 for the measurement-error model), **random seed 20260503**. Pre-specified convergence thresholds: rank-normalized split-R̂ ≤ 1.01 and bulk ESS > 400 for every monitored parameter. Total runtime ≈ 5–10 minutes on a standard laptop (4 cores).

## Data sources

All data are publicly available; the analysis-ready panels in `data/` permit direct replication.

| Dataset | Used for | URL |
|---------|----------|-----|
| V-Dem v15 (`v2smpolsoc`, `v2cacamps`, codelow/codehigh) | Outcome; SA4; SA3 | https://www.v-dem.net/data/the-v-dem-dataset/ |
| OWID excess mortality (HMD/STMF + World Mortality Dataset) | Mortality-burden exposure (P-score) | https://ourworldindata.org/excess-mortality-covid |
| OxCGRT Stringency Index | Policy-stringency exposure | https://github.com/OxCGRT/covid-policy-dataset |
| EM-DAT | Historical epidemic exposure (Question 3) | https://www.emdat.be/ |
| HIEF | Ethnic fractionalization covariate | (data/raw/hief.csv) |

Key variable: `p_score_mean` = country-year mean of monthly all-age excess-mortality P-scores. (Earlier repository iterations described a log-deaths-per-million variable, `covid_intensity`, from a superseded frequentist design; the fitted models use `p_score_mean` throughout.)

## Planned update

We commit to re-estimating the primary and sensitivity models as post-2024 V-Dem waves and mortality data accumulate, as a direct out-of-window test of the null short-run shock associations reported in the manuscript. Watch this repository for the tagged update.

## Citation

> Kawahara T, Fujiwara T. Pre-pandemic social cohesion was associated with COVID-19 lockdown stringency, but pandemic shocks did not shift polarization trajectories: a Bayesian dynamic panel analysis of 37 OECD countries. Manuscript under review (2026). The journal reference and DOI will be added upon publication.

## Contact & license

**Tomoki Kawahara** — Department of Public Health, Institute of Science Tokyo (kawahara.hlth@tmd.ac.jp).
Code: MIT License. Data products inherit the licenses of their upstream sources (V-Dem, OWID, OxCGRT, EM-DAT).
