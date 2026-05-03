#!/usr/bin/env python3
"""
EIPSA – COVID-19 Integration Analysis
=======================================
Tests whether COVID-19 mortality intensity is associated with
democratic erosion in OECD countries, using OWID COVID-19 data
merged with V-Dem v15 democratic indicators.

Data sources
------------
- OWID COVID-19:  owid-covid-data.csv  (daily, 2020–2024)
- V-Dem v15:      V-Dem-CY-Full+Others-v15.csv  (annual, up to 2024)
- EM-DAT:         pre-2020 epidemics (for combined panel, optional)

Requirements:
    pip install pandas openpyxl linearmodels matplotlib
"""

from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import textwrap
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 0.  Paths & constants ──────────────────────────────────────────────
OWID_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data/"
    "master/public/data/owid-covid-data.csv"
)
OWID_LOCAL = "./owid-covid-data.csv"

EMDAT_PATH = (
    "./public_emdat_custom_request_2026-02-11_"
    "567cb81c-dcaf-4e1c-9957-b38975a5a068.xlsx"
)
VDEM_PATH = "./V-Dem-CY-FullOthers-v15_csv/V-Dem-CY-Full+Others-v15.csv"

OUTCOMES = {
    "v2x_libdem": "Liberal Democracy Index",
    "v2x_jucon":  "Judicial Constraints on Executive",
    "v2x_civlib": "Civil Liberties Index",
}

OECD = sorted([
    "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE",
    "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "ISL",
    "IRL", "ISR", "ITA", "JPN", "KOR", "LVA", "LTU", "LUX",
    "MEX", "NLD", "NZL", "NOR", "POL", "PRT", "SVK", "SVN",
    "ESP", "SWE", "CHE", "TUR", "GBR", "USA",
])

LAGS = [1, 2, 3]  # post-shock lags for panel analysis


# =====================================================================
# HELPERS
# =====================================================================
def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "\u2020"
    return ""


def run_twfe(dep_var: str, exog_vars: list[str],
             data: pd.DataFrame, min_obs: int = 30) -> dict | None:
    """TWFE regression; returns stats for the FIRST exog variable."""
    cols = [dep_var] + exog_vars
    subset = data[cols].dropna()
    n_entities = subset.index.get_level_values(0).nunique()
    if subset.shape[0] < min_obs or n_entities < 3:
        return None

    y = subset[dep_var]
    x = subset[exog_vars]
    mod = PanelOLS(y, x, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    treat = exog_vars[0]
    return {
        "beta":       res.params[treat],
        "se":         res.std_errors[treat],
        "ci_lo":      res.conf_int().loc[treat, "lower"],
        "ci_hi":      res.conf_int().loc[treat, "upper"],
        "pval":       res.pvalues[treat],
        "nobs":       int(res.nobs),
        "n_entities": n_entities,
        "r2_within":  res.rsquared_within,
        "result_obj": res,
    }


def print_coef(label: str, info: dict | None) -> None:
    if info is None:
        print(f"  {label:>30s}:  insufficient observations")
        return
    p = info["pval"]
    print(
        f"  {label:>30s}:  \u03b2 = {info['beta']:+.6f} "
        f"(SE = {info['se']:.6f})  "
        f"95% CI [{info['ci_lo']:+.6f}, {info['ci_hi']:+.6f}]  "
        f"p = {p:.4f}{sig_stars(p)}  "
        f"N = {info['nobs']}  n_c = {info['n_entities']}"
    )


# =====================================================================
# 1.  LOAD & PROCESS OWID COVID-19 DATA
# =====================================================================
print("=" * 76)
print(" 1.  LOADING COVID-19 DATA (OWID)")
print("=" * 76)

if os.path.exists(OWID_LOCAL):
    print(f"  Loading from local file: {OWID_LOCAL}")
    owid_raw = pd.read_csv(OWID_LOCAL, low_memory=False)
else:
    print(f"  Downloading from GitHub \u2026")
    owid_raw = pd.read_csv(OWID_URL, low_memory=False)
    owid_raw.to_csv(OWID_LOCAL, index=False)
    print(f"  Saved to {OWID_LOCAL}")

print(f"  Raw rows: {len(owid_raw):,}")

# Parse dates, filter to real countries (exclude OWID_ aggregates)
owid_raw["date"] = pd.to_datetime(owid_raw["date"])
owid = owid_raw[~owid_raw["iso_code"].str.startswith("OWID_", na=True)].copy()
owid["year"] = owid["date"].dt.year

# Aggregate to country-year
owid_cy = (
    owid
    .groupby(["iso_code", "year"], as_index=False)
    .agg(
        total_deaths_pm_eoy=("total_deaths_per_million", "max"),
        total_cases_pm_eoy=("total_cases_per_million", "max"),
        new_deaths_pm_annual=("new_deaths_per_million", "sum"),
        stringency_mean=("stringency_index", "mean"),
        stringency_max=("stringency_index", "max"),
        population=("population", "max"),
    )
)
owid_cy = owid_cy.rename(columns={"iso_code": "iso3"})

# COVID intensity measures
owid_cy["covid_deaths_pm"] = owid_cy["total_deaths_pm_eoy"].fillna(0)
owid_cy["covid_intensity"] = np.log1p(owid_cy["covid_deaths_pm"])

# Year-on-year incremental deaths per million
owid_cy = owid_cy.sort_values(["iso3", "year"])
owid_cy["covid_deaths_pm_annual"] = (
    owid_cy.groupby("iso3")["covid_deaths_pm"].diff().fillna(
        owid_cy["covid_deaths_pm"]
    )
)
owid_cy["covid_annual_intensity"] = np.log1p(
    owid_cy["covid_deaths_pm_annual"].clip(lower=0)
)

print(f"  Country-years: {len(owid_cy):,}")
print(f"  Year range: {owid_cy['year'].min()}\u2013{owid_cy['year'].max()}")

# Show OECD summary
oecd_owid = owid_cy[owid_cy["iso3"].isin(OECD)]
print(f"\n  OECD COVID-19 summary (cumulative deaths/million by end of year):")
pivot = oecd_owid.pivot_table(
    index="iso3", columns="year", values="covid_deaths_pm", aggfunc="max"
)
print(pivot[[c for c in [2020, 2021, 2022, 2023] if c in pivot.columns]]
      .round(0).to_string())


# =====================================================================
# 2.  LOAD V-Dem
# =====================================================================
print(f"\n{'=' * 76}")
print(" 2.  LOADING V-Dem v15")
print("=" * 76)

vdem_cols = (
    ["country_text_id", "country_name", "year"]
    + list(OUTCOMES.keys())
    + ["e_gdppc"]
)
vdem = pd.read_csv(VDEM_PATH, usecols=vdem_cols, low_memory=False)
vdem = vdem.rename(columns={"country_text_id": "iso3"})
vdem = vdem.dropna(subset=list(OUTCOMES.keys()), how="all")
vdem["ln_gdppc"] = np.log(vdem["e_gdppc"].replace(0, np.nan))
print(f"  Rows: {len(vdem):,}")


# =====================================================================
# 3.  MERGE & CONSTRUCT PANEL
# =====================================================================
print(f"\n{'=' * 76}")
print(" 3.  MERGING V-Dem + OWID COVID")
print("=" * 76)

# ── 3a. Merge V-Dem with OWID ──
panel = vdem.merge(
    owid_cy[["iso3", "year", "covid_deaths_pm", "covid_intensity",
             "covid_deaths_pm_annual", "covid_annual_intensity",
             "stringency_mean", "stringency_max"]],
    on=["iso3", "year"],
    how="left",
)

# For years without COVID data (pre-2020), fill with 0
for col in ["covid_deaths_pm", "covid_intensity",
            "covid_deaths_pm_annual", "covid_annual_intensity",
            "stringency_mean", "stringency_max"]:
    panel[col] = panel[col].fillna(0)

# ── 3b. Optionally merge EM-DAT pre-2020 epidemics ──
if os.path.exists(EMDAT_PATH):
    print("  Merging EM-DAT pre-2020 epidemics \u2026")
    emdat_raw = pd.read_excel(EMDAT_PATH)
    emdat = emdat_raw[emdat_raw["Disaster Type"] == "Epidemic"].copy()
    emdat = emdat.rename(columns={"ISO": "iso3", "Start Year": "year"})
    emdat["Total Deaths"] = pd.to_numeric(
        emdat["Total Deaths"], errors="coerce"
    ).fillna(0)
    emdat_cy = (
        emdat
        .groupby(["iso3", "year"], as_index=False)
        .agg(emdat_deaths=("Total Deaths", "sum"))
    )
    emdat_cy["emdat_epidemic"] = 1

    panel = panel.merge(emdat_cy, on=["iso3", "year"], how="left")
    panel["emdat_epidemic"] = panel["emdat_epidemic"].fillna(0).astype(int)
    panel["emdat_deaths"] = panel["emdat_deaths"].fillna(0)

    # Combined "pandemic fear" variable: EM-DAT pre-2020, COVID 2020+
    panel["combined_intensity"] = np.where(
        panel["year"] < 2020,
        np.log1p(panel["emdat_deaths"]),   # pre-2020: EM-DAT deaths (raw)
        panel["covid_annual_intensity"],    # 2020+: COVID deaths/million
    )
else:
    print("  EM-DAT file not found; using COVID-only exposure.")
    panel["combined_intensity"] = panel["covid_annual_intensity"]

# ── 3c. First-difference outcomes ──
panel = panel.sort_values(["iso3", "year"])
for v in OUTCOMES:
    panel[f"d_{v}"] = panel.groupby("iso3")[v].diff()

# ── 3d. Lag COVID exposure ──
for k in LAGS:
    panel[f"covid_intensity_L{k}"] = (
        panel.groupby("iso3")["covid_annual_intensity"].shift(k)
    )

# ── 3e. Set panel index & OECD subset ──
panel = panel.set_index(["iso3", "year"])
panel_oecd = panel.loc[panel.index.get_level_values(0).isin(OECD)].copy()

# Restrict to modern era with sufficient democracy variation
panel_oecd_modern = panel_oecd.loc[
    panel_oecd.index.get_level_values("year") >= 2000
].copy()

n_oecd = panel_oecd_modern.index.get_level_values(0).nunique()
print(f"\n  OECD panel (2000\u20132024): {len(panel_oecd_modern):,} obs, "
      f"{n_oecd} countries")

# Descriptive: COVID intensity across OECD
covid_oecd = panel_oecd_modern[
    panel_oecd_modern["covid_annual_intensity"] > 0
]
print(f"  OECD country-years with COVID deaths: {len(covid_oecd)}")
print(f"  covid_annual_intensity range: "
      f"{covid_oecd['covid_annual_intensity'].min():.2f} \u2013 "
      f"{covid_oecd['covid_annual_intensity'].max():.2f}")


# =====================================================================
# 4.  PANEL REGRESSIONS — OECD, COVID-19 EXPOSURE
# =====================================================================
print(f"\n\n{'=' * 76}")
print(" 4.  PANEL REGRESSIONS \u2014 OECD countries, 2000\u20132024")
print("=" * 76)

lag_labels = {
    "covid_annual_intensity": "t (contemp.)",
    **{f"covid_intensity_L{k}": f"t\u2013{k}" for k in LAGS},
}
all_exposure_vars = list(lag_labels.keys())

# ── 4a. Level outcomes: Y_it ──
print(f"\n{'─' * 76}")
print(" A. Level outcomes: Y_it = \u03b1_i + \u03b3_t + \u03b2 \u00b7 COVID_it + \u03b5")
print(f"{'─' * 76}")

for outcome, olabel in OUTCOMES.items():
    print(f"\n  {outcome}  ({olabel})")
    for evar in all_exposure_vars:
        label = lag_labels[evar]
        info = run_twfe(outcome, [evar], panel_oecd_modern)
        print_coef(f"COVID intensity {label}", info)
        if info and evar == "covid_annual_intensity" and outcome == "v2x_libdem":
            print("\n  Full summary:")
            print(info["result_obj"].summary)

# ── 4b. First-differenced: ΔY_it ──
print(f"\n{'─' * 76}")
print(" B. First-differenced: \u0394Y_it = \u03b1_i + \u03b3_t + \u03b2 \u00b7 COVID_it + \u03b5")
print("    (\u03b2 < 0 \u2192 COVID associated with democratic decline)")
print(f"{'─' * 76}")

d_outcomes = {f"d_{v}": f"\u0394 {lab}" for v, lab in OUTCOMES.items()}

fd_results = {}
for dv, dlabel in d_outcomes.items():
    print(f"\n  {dv}  ({dlabel})")
    for evar in all_exposure_vars:
        label = lag_labels[evar]
        info = run_twfe(dv, [evar], panel_oecd_modern)
        fd_results[(dv, label)] = info
        print_coef(f"COVID intensity {label}", info)
        if info and evar == "covid_annual_intensity" and dv == "d_v2x_libdem":
            print("\n  Full summary:")
            print(info["result_obj"].summary)

# ── 4c. With stringency control ──
print(f"\n{'─' * 76}")
print(" C. First-differenced + Stringency Index control")
print("    (isolates COVID fear from policy response)")
print(f"{'─' * 76}")

for dv, dlabel in d_outcomes.items():
    print(f"\n  {dv}  ({dlabel})")
    for evar in all_exposure_vars:
        label = lag_labels[evar]
        info = run_twfe(dv, [evar, "stringency_mean"], panel_oecd_modern)
        print_coef(f"COVID + stringency {label}", info)


# =====================================================================
# 5.  CROSS-SECTIONAL SCATTER PLOT
# =====================================================================
print(f"\n\n{'=' * 76}")
print(" 5.  CROSS-SECTIONAL ANALYSIS \u2014 COVID shock vs. democracy change")
print("=" * 76)

# Build cross-section: one row per OECD country
vdem_unstacked = vdem[vdem["iso3"].isin(OECD)].copy()

cs = pd.DataFrame({"iso3": OECD})

# COVID deaths per million (cumulative through end of 2021)
covid_cum_2021 = (
    owid_cy[(owid_cy["iso3"].isin(OECD)) & (owid_cy["year"] == 2021)]
    [["iso3", "covid_deaths_pm"]]
)
cs = cs.merge(covid_cum_2021, on="iso3", how="left")
cs = cs.rename(columns={"covid_deaths_pm": "covid_deaths_pm_2021"})

# Democracy index at 2019 (pre-COVID baseline) and 2022 / 2023
for yr in [2019, 2022, 2023, 2024]:
    yr_data = vdem_unstacked[vdem_unstacked["year"] == yr][
        ["iso3", "v2x_libdem"]
    ].rename(columns={"v2x_libdem": f"libdem_{yr}"})
    cs = cs.merge(yr_data, on="iso3", how="left")

# Main outcome: change from 2019 to 2022
cs["d_libdem_19_22"] = cs["libdem_2022"] - cs["libdem_2019"]
# Alternative: 2019 to 2023
cs["d_libdem_19_23"] = cs["libdem_2023"] - cs["libdem_2019"]
# Extended: 2019 to 2024
cs["d_libdem_19_24"] = cs["libdem_2024"] - cs["libdem_2019"]

cs["ln_covid_deaths"] = np.log1p(cs["covid_deaths_pm_2021"].fillna(0))

# Country names for labelling
names = vdem_unstacked.drop_duplicates("iso3")[["iso3", "country_name"]]
cs = cs.merge(names, on="iso3", how="left")

print("\n  Cross-sectional data (38 OECD countries):")
print(cs[["iso3", "country_name", "covid_deaths_pm_2021",
          "libdem_2019", "libdem_2022", "d_libdem_19_22"]]
      .sort_values("covid_deaths_pm_2021", ascending=False)
      .to_string(index=False))

# ── OLS on the cross-section ──
from scipy import stats as sp_stats

for dep, dep_label, color in [
    ("d_libdem_19_22", "\u0394 LibDem (2019\u21922022)", "#1f77b4"),
    ("d_libdem_19_23", "\u0394 LibDem (2019\u21922023)", "#d62728"),
    ("d_libdem_19_24", "\u0394 LibDem (2019\u21922024)", "#2ca02c"),
]:
    valid = cs.dropna(subset=["ln_covid_deaths", dep])
    slope, intercept, r, p, se = sp_stats.linregress(
        valid["ln_covid_deaths"], valid[dep]
    )
    print(f"\n  OLS: {dep_label}")
    print(f"    slope = {slope:+.5f}  (SE = {se:.5f})  "
          f"r = {r:.3f}  R\u00b2 = {r**2:.3f}  p = {p:.4f}{sig_stars(p)}")

# ── Scatter plots ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (dep, dep_label) in zip(axes, [
    ("d_libdem_19_22", "\u0394 LibDem (2019 \u2192 2022)"),
    ("d_libdem_19_23", "\u0394 LibDem (2019 \u2192 2023)"),
    ("d_libdem_19_24", "\u0394 LibDem (2019 \u2192 2024)"),
]):
    valid = cs.dropna(subset=["ln_covid_deaths", dep])

    ax.scatter(
        valid["ln_covid_deaths"], valid[dep],
        s=40, alpha=0.7, edgecolors="k", linewidths=0.5, zorder=3,
    )

    # Label notable countries
    highlight = ["USA", "GBR", "HUN", "TUR", "POL", "MEX",
                 "NZL", "AUS", "JPN", "CHL", "ITA", "ESP",
                 "KOR", "ISR", "CZE", "SVN", "COL"]
    for _, row in valid.iterrows():
        if row["iso3"] in highlight:
            ax.annotate(
                row["iso3"],
                (row["ln_covid_deaths"], row[dep]),
                fontsize=7, ha="left", va="bottom",
                xytext=(4, 3), textcoords="offset points",
            )

    # Regression line
    slope, intercept, r, p, se = sp_stats.linregress(
        valid["ln_covid_deaths"], valid[dep]
    )
    x_line = np.linspace(
        valid["ln_covid_deaths"].min(), valid["ln_covid_deaths"].max(), 100
    )
    ax.plot(x_line, intercept + slope * x_line,
            "r--", linewidth=1.5, alpha=0.8)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("ln(1 + cumulative COVID deaths per million, end 2021)")
    ax.set_ylabel(dep_label)
    ax.set_title(
        f"{dep_label}\n"
        f"slope = {slope:+.4f}, R\u00b2 = {r**2:.3f}, p = {p:.4f}",
        fontsize=10,
    )

fig.suptitle(
    "OECD countries: COVID-19 death toll vs. change in Liberal Democracy\n"
    "(V-Dem v2x_libdem; each dot = one country)",
    fontsize=12, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig("eipsa_covid_scatter.png", dpi=200, bbox_inches="tight")
print(f"\nScatter plot saved \u2192 eipsa_covid_scatter.png")
plt.close()


# =====================================================================
# 6.  ROBUSTNESS: within-country COVID variation over time
# =====================================================================
print(f"\n\n{'=' * 76}")
print(" 6.  WITHIN-COUNTRY COVID WAVE ANALYSIS")
print("=" * 76)
print("  Using year-by-year COVID death intensity (2020\u20132024) as exposure")
print("  within OECD countries that vary over time.\n")

# Restrict to 2019-2024 (need 2019 for first-difference into 2020)
panel_covid_era = panel_oecd.loc[
    panel_oecd.index.get_level_values("year").isin(range(2019, 2025))
].copy()

print(f"  Panel: {len(panel_covid_era)} obs, "
      f"{panel_covid_era.index.get_level_values(0).nunique()} countries, "
      f"2019\u20132024")

for dv, dlabel in d_outcomes.items():
    info = run_twfe(dv, ["covid_annual_intensity"], panel_covid_era)
    print_coef(f"{dlabel} ~ COVID intensity", info)
    if info and dv == "d_v2x_libdem":
        print(info["result_obj"].summary)


# =====================================================================
# 7.  COMBINED TIMELINE PLOT
# =====================================================================
print(f"\n{'=' * 76}")
print(" 7.  DESCRIPTIVE TIMELINE")
print("=" * 76)

# Compute OECD-wide means by year
oecd_ts = (
    panel_oecd
    .loc[panel_oecd.index.get_level_values("year") >= 2000]
    .reset_index()
    .groupby("year")
    .agg(
        libdem_mean=("v2x_libdem", "mean"),
        libdem_sd=("v2x_libdem", "std"),
        covid_mean=("covid_annual_intensity", "mean"),
    )
)

fig, ax1 = plt.subplots(figsize=(10, 5))

color_dem = "#1f77b4"
color_covid = "#d62728"

ax1.set_xlabel("Year")
ax1.set_ylabel("Mean v2x_libdem (OECD)", color=color_dem)
ax1.fill_between(
    oecd_ts.index,
    oecd_ts["libdem_mean"] - oecd_ts["libdem_sd"],
    oecd_ts["libdem_mean"] + oecd_ts["libdem_sd"],
    alpha=0.15, color=color_dem,
)
ax1.plot(oecd_ts.index, oecd_ts["libdem_mean"],
         "o-", color=color_dem, linewidth=2, markersize=4, label="LibDem")
ax1.tick_params(axis="y", labelcolor=color_dem)

ax2 = ax1.twinx()
ax2.set_ylabel("Mean COVID intensity ln(1+deaths/M)", color=color_covid)
ax2.bar(oecd_ts.index, oecd_ts["covid_mean"],
        color=color_covid, alpha=0.4, width=0.7, label="COVID intensity")
ax2.tick_params(axis="y", labelcolor=color_covid)

ax1.axvspan(2019.5, 2024.5, alpha=0.08, color="red", label="COVID era")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)

ax1.set_title(
    "OECD mean: Liberal Democracy Index vs. COVID-19 intensity\n"
    "(V-Dem v2x_libdem; OWID deaths per million)",
    fontsize=11, fontweight="bold",
)
fig.tight_layout()
fig.savefig("eipsa_covid_timeline.png", dpi=200, bbox_inches="tight")
print(f"Timeline plot saved \u2192 eipsa_covid_timeline.png")
plt.close()


# =====================================================================
# 8.  DIAGNOSTICS & SUMMARY
# =====================================================================
print(f"\n\n{'=' * 76}")
print(" 8.  SUMMARY & CAVEATS")
print("=" * 76)

print(textwrap.dedent("""
  DATA COVERAGE:
    OWID COVID-19:   2020-01 to 2024-08  (daily \u2192 annual aggregation)
    V-Dem v15:       up to 2024           (38/38 OECD countries)
    Exposure:        covid_annual_intensity = ln(1 + annual deaths per million)

  KEY RESULTS SUMMARY:
"""))

# Collect headline numbers
for dv, dlabel in d_outcomes.items():
    key = (dv, "t (contemp.)")
    r = fd_results.get(key)
    if r:
        print(f"    {dlabel:>40s}:  \u03b2 = {r['beta']:+.6f}  "
              f"p = {r['pval']:.4f}{sig_stars(r['pval'])}")
    else:
        print(f"    {dlabel:>40s}:  not estimated")

print(textwrap.dedent("""
  CAVEATS:
  1. V-Dem indices are expert-coded annual assessments, not survey-based.
     Year-to-year changes in OECD democracies are very small (order 0.01),
     so even a statistically significant coefficient may reflect expert
     rating adjustments rather than genuine institutional change.
  2. COVID-19 is a GLOBAL shock hitting all OECD countries simultaneously.
     With country + year fixed effects, identification relies on
     cross-country variation in death rates within the same year.
     Year FE absorb much of the common COVID shock.
  3. The cross-sectional scatter (Section 5) does NOT control for country
     fixed effects and is therefore more descriptive than causal.
  4. Reverse causality is possible: countries with weaker democratic
     institutions may have had worse pandemic responses and higher death
     rates (state capacity \u2192 deaths, rather than deaths \u2192 authoritarianism).
"""))

print("Done.")
