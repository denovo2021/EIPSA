#!/usr/bin/env python3
"""
EIPSA – Epidemic-Induced Political Shift Analysis  (v3)
========================================================
Two-Way Fixed Effects (TWFE) estimation of the relationship between
local epidemic outbreaks and subsequent changes in democratic governance
indicators, using EM-DAT (exposure) and V-Dem v15 (outcome) data.

v3 — Developed-nation "fear mechanism" specification
-----------------------------------------------------
- Sub-sample restricted to OECD-38 countries
- Exposure: binary (any epidemic), major-epidemic (deaths > 10),
  and continuous ln(1 + deaths)
- Outcome: first-differenced (ΔY = Y_t − Y_{t−1}) to capture
  *acceleration* of democratic change, not level differences
- Event-study with leads for pre-trends validation
- Full-sample results retained as benchmark

NOTE: This EM-DAT extract does NOT contain COVID-19 records.
      COVID-19 is coded separately by EM-DAT and was not included
      in the custom download.  Results therefore reflect non-COVID
      epidemics only (cholera, dengue, MERS, SARS, H5N1, etc.).

Requirements:
    pip install pandas openpyxl linearmodels matplotlib
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import textwrap
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 0.  Configuration ──────────────────────────────────────────────────
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

LAGS_SIMPLE = [1, 3, 5]

# Event-study window
ES_LEADS = [2, 1]
ES_LAGS  = [0, 1, 2, 3]

# Major-epidemic threshold (deaths in a country-year)
MAJOR_THRESHOLD = 10

# OECD-38 member states (as of 2024)
OECD = [
    "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE",
    "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "ISL",
    "IRL", "ISR", "ITA", "JPN", "KOR", "LVA", "LTU", "LUX",
    "MEX", "NLD", "NZL", "NOR", "POL", "PRT", "SVK", "SVN",
    "ESP", "SWE", "CHE", "TUR", "GBR", "USA",
]


# =====================================================================
# HELPER
# =====================================================================
def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "\u2020"
    return ""


def run_twfe(dep_var: str, exog_vars: list[str],
             data: pd.DataFrame, min_obs: int = 60) -> dict | None:
    """TWFE regression; returns stats for the FIRST exog variable."""
    cols = [dep_var] + exog_vars
    subset = data[cols].dropna()
    if subset.shape[0] < min_obs:
        return None
    n_entities = subset.index.get_level_values(0).nunique()
    if n_entities < 3:
        return None

    y = subset[dep_var]
    x = subset[exog_vars]
    mod = PanelOLS(y, x, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    treat = exog_vars[0]
    return {
        "beta":      res.params[treat],
        "se":        res.std_errors[treat],
        "ci_lo":     res.conf_int().loc[treat, "lower"],
        "ci_hi":     res.conf_int().loc[treat, "upper"],
        "pval":      res.pvalues[treat],
        "nobs":      res.nobs,
        "n_entities": n_entities,
        "r2_within": res.rsquared_within,
        "result_obj": res,
    }


def print_row(label: str, info: dict | None) -> None:
    if info is None:
        print(f"  {label:>20s}:  insufficient observations – skipped")
        return
    p = info["pval"]
    print(
        f"  {label:>20s}:  \u03b2 = {info['beta']:+.5f} "
        f"(SE = {info['se']:.5f})  "
        f"95% CI [{info['ci_lo']:+.5f}, {info['ci_hi']:+.5f}]  "
        f"p = {p:.4f}{sig_stars(p)}  "
        f"N = {info['nobs']:,}  "
        f"n_c = {info['n_entities']}  "
        f"R\u00b2(w) = {info['r2_within']:.4f}"
    )


# =====================================================================
# 1.  DATA LOADING & CLEANING
# =====================================================================
print("Loading EM-DAT \u2026")
emdat_raw = pd.read_excel(EMDAT_PATH)
emdat = emdat_raw[emdat_raw["Disaster Type"] == "Epidemic"].copy()
print(f"  Epidemic records: {len(emdat):,}")

emdat = emdat.rename(columns={"ISO": "iso3", "Start Year": "year"})
for col in ["Total Deaths", "Total Affected"]:
    emdat[col] = pd.to_numeric(emdat[col], errors="coerce").fillna(0)

emdat_cy = (
    emdat
    .groupby(["iso3", "year"], as_index=False)
    .agg(
        epidemic_deaths=("Total Deaths", "sum"),
        epidemic_affected=("Total Affected", "sum"),
        epidemic_count=("Disaster Type", "size"),
    )
)
emdat_cy["epidemic"] = 1
emdat_cy["major_epidemic"] = (
    emdat_cy["epidemic_deaths"] > MAJOR_THRESHOLD
).astype(int)

print(f"  Country-years with \u22651 epidemic: {len(emdat_cy):,}")
print(f"  Country-years with major epidemic (deaths > {MAJOR_THRESHOLD}): "
      f"{emdat_cy['major_epidemic'].sum():,}")

# ── V-Dem ──
print("Loading V-Dem v15 \u2026")
vdem_cols = (
    ["country_text_id", "country_name", "year"]
    + list(OUTCOMES.keys())
    + ["e_gdppc"]
)
vdem = pd.read_csv(VDEM_PATH, usecols=vdem_cols, low_memory=False)
vdem = vdem.rename(columns={"country_text_id": "iso3"})
vdem = vdem.dropna(subset=list(OUTCOMES.keys()), how="all")
vdem["ln_gdppc"] = np.log(vdem["e_gdppc"].replace(0, np.nan))
print(f"  V-Dem rows: {len(vdem):,}")

# ── Merge ──
print("Merging \u2026")
panel = vdem.merge(emdat_cy, on=["iso3", "year"], how="left")
panel["epidemic"] = panel["epidemic"].fillna(0).astype(int)
panel["major_epidemic"] = panel["major_epidemic"].fillna(0).astype(int)
for col in ["epidemic_deaths", "epidemic_affected", "epidemic_count"]:
    panel[col] = panel[col].fillna(0)

panel["ln_epi_deaths"] = np.log1p(panel["epidemic_deaths"])

print(f"  Full panel: {len(panel):,} country-years, "
      f"{panel['iso3'].nunique()} countries")

# ── Sort & first-difference outcomes ──
panel = panel.sort_values(["iso3", "year"])
for v in OUTCOMES:
    panel[f"d_{v}"] = panel.groupby("iso3")[v].diff()

# ── Leads & lags ──
for k in LAGS_SIMPLE:
    panel[f"epidemic_L{k}"] = panel.groupby("iso3")["epidemic"].shift(k)
    panel[f"major_epidemic_L{k}"] = panel.groupby("iso3")["major_epidemic"].shift(k)
    panel[f"ln_epi_deaths_L{k}"] = panel.groupby("iso3")["ln_epi_deaths"].shift(k)

for k in ES_LEADS:
    panel[f"epidemic_F{k}"] = panel.groupby("iso3")["epidemic"].shift(-k)
    panel[f"major_epidemic_F{k}"] = panel.groupby("iso3")["major_epidemic"].shift(-k)
    panel[f"ln_epi_deaths_F{k}"] = panel.groupby("iso3")["ln_epi_deaths"].shift(-k)
for k in ES_LAGS:
    if k == 0:
        continue
    # L1, L2, L3 already created for simple lags; L2 needs explicit creation
    if f"epidemic_L{k}" not in panel.columns:
        panel[f"epidemic_L{k}"] = panel.groupby("iso3")["epidemic"].shift(k)
    if f"major_epidemic_L{k}" not in panel.columns:
        panel[f"major_epidemic_L{k}"] = panel.groupby("iso3")["major_epidemic"].shift(k)
    if f"ln_epi_deaths_L{k}" not in panel.columns:
        panel[f"ln_epi_deaths_L{k}"] = panel.groupby("iso3")["ln_epi_deaths"].shift(k)

# ── Set panel index ──
panel = panel.set_index(["iso3", "year"])

# ── Build sub-panels ──
panel_oecd = panel.loc[panel.index.get_level_values(0).isin(OECD)].copy()

n_oecd = panel_oecd.index.get_level_values(0).nunique()
n_epi_oecd = int(panel_oecd["epidemic"].sum())
n_major_oecd = int(panel_oecd["major_epidemic"].sum())
print(f"\n  OECD sub-panel: {len(panel_oecd):,} obs, "
      f"{n_oecd} countries, "
      f"{panel_oecd.index.get_level_values(1).min()}\u2013"
      f"{panel_oecd.index.get_level_values(1).max()}")
print(f"  Epidemic country-years in OECD:       {n_epi_oecd}")
print(f"  Major epidemic (deaths>{MAJOR_THRESHOLD}) in OECD: "
      f"{n_major_oecd}")

# Show the actual OECD epidemic events for transparency
oecd_treated = panel_oecd[panel_oecd["epidemic"] == 1][
    ["epidemic_deaths", "epidemic_affected", "major_epidemic"]
]
print(f"\n  OECD treated observations ({len(oecd_treated)} country-years):")
print(oecd_treated.to_string())


# =====================================================================
# 2.  FULL-SAMPLE BENCHMARK  (levels, all countries)
# =====================================================================
lag_vars = ["epidemic"] + [f"epidemic_L{k}" for k in LAGS_SIMPLE]
lag_labels = {
    "epidemic": "t (contemp.)",
    **{f"epidemic_L{k}": f"t\u2013{k}" for k in LAGS_SIMPLE},
}

print(f"\n\n{'=' * 76}")
print(" BENCHMARK: Full sample, level outcomes, binary epidemic")
print(f"{'=' * 76}")

benchmark_store = {}
for outcome, olabel in OUTCOMES.items():
    print(f"\n  {outcome}  ({olabel})")
    for lvar in lag_vars:
        ll = lag_labels[lvar]
        info = run_twfe(outcome, [lvar], panel)
        benchmark_store[(outcome, ll)] = info
        print_row(f"Epidemic {ll}", info)


# =====================================================================
# 3.  OECD ANALYSIS — LEVEL OUTCOMES
# =====================================================================
print(f"\n\n{'=' * 76}")
print(f" OECD-ONLY: Level outcomes Y_it  (N countries = {n_oecd})")
print(f"{'=' * 76}")

oecd_level_store = {}

# 3a. Binary epidemic
print(f"\n{'─' * 76}")
print(" Exposure: binary epidemic (any)")
print(f"{'─' * 76}")
for outcome, olabel in OUTCOMES.items():
    print(f"\n  {outcome}  ({olabel})")
    for lvar in lag_vars:
        ll = lag_labels[lvar]
        info = run_twfe(outcome, [lvar], panel_oecd)
        oecd_level_store[("binary", outcome, ll)] = info
        print_row(f"Epidemic {ll}", info)

# 3b. Major epidemic (deaths > threshold)
major_vars = ["major_epidemic"] + [f"major_epidemic_L{k}" for k in LAGS_SIMPLE]
print(f"\n{'─' * 76}")
print(f" Exposure: major epidemic (deaths > {MAJOR_THRESHOLD})")
print(f"{'─' * 76}")
for outcome, olabel in OUTCOMES.items():
    print(f"\n  {outcome}  ({olabel})")
    for lvar, ll in zip(major_vars, lag_labels.values()):
        info = run_twfe(outcome, [lvar], panel_oecd)
        oecd_level_store[("major", outcome, ll)] = info
        print_row(f"Major epi {ll}", info)

# 3c. Continuous: ln(1 + deaths)
cont_vars = ["ln_epi_deaths"] + [f"ln_epi_deaths_L{k}" for k in LAGS_SIMPLE]
print(f"\n{'─' * 76}")
print(" Exposure: ln(1 + epidemic deaths)")
print(f"{'─' * 76}")
for outcome, olabel in OUTCOMES.items():
    print(f"\n  {outcome}  ({olabel})")
    for lvar, ll in zip(cont_vars, lag_labels.values()):
        info = run_twfe(outcome, [lvar], panel_oecd)
        oecd_level_store[("cont", outcome, ll)] = info
        print_row(f"ln(deaths) {ll}", info)


# =====================================================================
# 4.  OECD ANALYSIS — FIRST-DIFFERENCED OUTCOMES  (ΔY_it)
# =====================================================================
print(f"\n\n{'=' * 76}")
print(f" OECD-ONLY: First-differenced outcomes \u0394Y_it = Y_it - Y_{{it-1}}")
print(f"{'=' * 76}")
print("  Interpretation: \u03b2 > 0 means epidemic is associated with an")
print("  INCREASE in the index; \u03b2 < 0 means a DECREASE (authoritarian shift).")

oecd_fd_store = {}
d_outcomes = {f"d_{v}": f"\u0394 {lab}" for v, lab in OUTCOMES.items()}

# 4a. Binary
print(f"\n{'─' * 76}")
print(" Exposure: binary epidemic")
print(f"{'─' * 76}")
for dv, dlabel in d_outcomes.items():
    print(f"\n  {dv}  ({dlabel})")
    for lvar in lag_vars:
        ll = lag_labels[lvar]
        info = run_twfe(dv, [lvar], panel_oecd)
        oecd_fd_store[("binary", dv, ll)] = info
        print_row(f"Epidemic {ll}", info)

# 4b. Major epidemic
print(f"\n{'─' * 76}")
print(f" Exposure: major epidemic (deaths > {MAJOR_THRESHOLD})")
print(f"{'─' * 76}")
for dv, dlabel in d_outcomes.items():
    print(f"\n  {dv}  ({dlabel})")
    for lvar, ll in zip(major_vars, lag_labels.values()):
        info = run_twfe(dv, [lvar], panel_oecd)
        oecd_fd_store[("major", dv, ll)] = info
        print_row(f"Major epi {ll}", info)

# 4c. Continuous
print(f"\n{'─' * 76}")
print(" Exposure: ln(1 + epidemic deaths)")
print(f"{'─' * 76}")
for dv, dlabel in d_outcomes.items():
    print(f"\n  {dv}  ({dlabel})")
    for lvar, ll in zip(cont_vars, lag_labels.values()):
        info = run_twfe(dv, [lvar], panel_oecd)
        oecd_fd_store[("cont", dv, ll)] = info
        print_row(f"ln(deaths) {ll}", info)


# =====================================================================
# 5.  EVENT STUDY — OECD, first-differenced v2x_libdem
# =====================================================================
print(f"\n\n{'=' * 76}")
print(" EVENT STUDY — OECD, \u0394v2x_libdem")
print(f"{'=' * 76}")

# Build var lists for each exposure type
exposure_types = {
    "binary":  ("epidemic",       "epidemic_F",       "epidemic_L"),
    "major":   ("major_epidemic", "major_epidemic_F", "major_epidemic_L"),
    "ln_deaths": ("ln_epi_deaths","ln_epi_deaths_F", "ln_epi_deaths_L"),
}

es_positions = [-k for k in ES_LEADS] + [0] + [k for k in ES_LAGS if k > 0]
es_tick = [f"t\u2013{k}" for k in ES_LEADS] + ["t"] + \
          [f"t+{k}" for k in ES_LAGS if k > 0]

es_all = {}  # {exposure_type: {dep_var: {var_name: coef_dict}}}

for etype, (base, fpfx, lpfx) in exposure_types.items():
    es_vars = (
        [f"{fpfx}{k}" for k in ES_LEADS]
        + [base]
        + [f"{lpfx}{k}" for k in ES_LAGS if k > 0]
    )

    es_all[etype] = {}
    for dv in [f"d_{v}" for v in OUTCOMES]:
        cols = [dv] + es_vars
        subset = panel_oecd[cols].dropna()
        if subset.shape[0] < 60 or subset.index.get_level_values(0).nunique() < 3:
            print(f"  {etype} / {dv}: insufficient obs ({len(subset)}) – skipped")
            continue

        y = subset[dv]
        x = subset[es_vars]
        mod = PanelOLS(y, x, entity_effects=True, time_effects=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        if dv == "d_v2x_libdem":
            print(f"\n  [{etype}] \u0394v2x_libdem")
            print(res.summary)

        es_all[etype][dv] = {}
        for var in es_vars:
            es_all[etype][dv][var] = {
                "beta":  res.params[var],
                "se":    res.std_errors[var],
                "ci_lo": res.conf_int().loc[var, "lower"],
                "ci_hi": res.conf_int().loc[var, "upper"],
                "pval":  res.pvalues[var],
            }


# ── Event-study plot (OECD, ΔY) ──
fig, axes = plt.subplots(
    len(exposure_types), len(OUTCOMES),
    figsize=(5.5 * len(OUTCOMES), 4.5 * len(exposure_types)),
    sharey=False,
)
if len(OUTCOMES) == 1:
    axes = axes.reshape(-1, 1)

for row, (etype, (base, fpfx, lpfx)) in enumerate(exposure_types.items()):
    es_vars = (
        [f"{fpfx}{k}" for k in ES_LEADS]
        + [base]
        + [f"{lpfx}{k}" for k in ES_LAGS if k > 0]
    )
    for col, (outcome, olabel) in enumerate(OUTCOMES.items()):
        ax = axes[row, col]
        dv = f"d_{outcome}"

        if etype not in es_all or dv not in es_all[etype]:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="grey")
            ax.set_title(f"{olabel}\n[{etype}]", fontsize=9)
            continue

        betas, ci_los, ci_his = [], [], []
        for var in es_vars:
            r = es_all[etype][dv][var]
            betas.append(r["beta"])
            ci_los.append(r["ci_lo"])
            ci_his.append(r["ci_hi"])

        betas   = np.array(betas)
        ci_los  = np.array(ci_los)
        ci_his  = np.array(ci_his)

        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.axvline(-0.5, color="red", linewidth=0.8, linestyle=":",
                   label="Treatment onset")
        ax.errorbar(
            es_positions, betas,
            yerr=[betas - ci_los, ci_his - betas],
            fmt="s-", capsize=4, color="#1f77b4", linewidth=1.5, markersize=6,
        )
        ax.set_xticks(es_positions)
        ax.set_xticklabels(es_tick)
        if row == len(exposure_types) - 1:
            ax.set_xlabel("Relative time")
        ax.set_ylabel(f"\u03b2  (\u0394{outcome})")
        ax.set_title(f"\u0394 {olabel}\n[{etype}]", fontsize=9,
                     fontweight="bold")
        if row == 0 and col == 0:
            ax.legend(fontsize=7, loc="best")

fig.suptitle(
    "Event study \u2014 OECD countries, first-differenced outcomes\n"
    "(TWFE; entity & time FE; clustered SEs; 95% CI)\n"
    "NOTE: EM-DAT extract excludes COVID-19",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig("eipsa_oecd_event_study.png", dpi=200, bbox_inches="tight")
print(f"\nOECD event-study plot saved \u2192 eipsa_oecd_event_study.png")
plt.close()


# =====================================================================
# 6.  COMPARISON PLOT — Full sample vs OECD  (ΔY, binary)
# =====================================================================
# Run full-sample first-differenced for comparison
fullsample_fd = {}
for dv in d_outcomes:
    for lvar in lag_vars:
        ll = lag_labels[lvar]
        fullsample_fd[(dv, ll)] = run_twfe(dv, [lvar], panel)

fig, axes = plt.subplots(1, len(OUTCOMES),
                         figsize=(5.5 * len(OUTCOMES), 5), sharey=False)
if len(OUTCOMES) == 1:
    axes = [axes]

lag_pos = [0] + LAGS_SIMPLE
lag_tick = ["t"] + [f"t+{k}" for k in LAGS_SIMPLE]
all_ll = list(lag_labels.values())

for ax, (outcome, olabel) in zip(axes, OUTCOMES.items()):
    dv = f"d_{outcome}"
    for store, clr, lbl, offset in [
        (fullsample_fd, "#1f77b4", f"All countries (N\u2248200)", -0.15),
        (oecd_fd_store,  "#d62728", f"OECD-38 only",              +0.15),
    ]:
        betas, ci_los, ci_his, positions = [], [], [], []
        for pos, ll in zip(lag_pos, all_ll):
            if store is oecd_fd_store:
                key = ("binary", dv, ll)
            else:
                key = (dv, ll)
            r = store.get(key)
            if r is None:
                continue
            betas.append(r["beta"])
            ci_los.append(r["ci_lo"])
            ci_his.append(r["ci_hi"])
            positions.append(pos + offset)

        betas   = np.array(betas)
        ci_los  = np.array(ci_los)
        ci_his  = np.array(ci_his)

        if len(betas) > 0:
            ax.errorbar(
                positions, betas,
                yerr=[betas - ci_los, ci_his - betas],
                fmt="o-", capsize=4, color=clr, linewidth=1.5,
                markersize=6, label=lbl,
            )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xticks(lag_pos)
    ax.set_xticklabels(lag_tick)
    ax.set_xlabel("Epidemic timing (lag)")
    ax.set_ylabel(f"\u03b2  (\u0394{outcome})")
    ax.set_title(f"\u0394 {olabel}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="best")

fig.suptitle(
    "TWFE: first-differenced outcomes, binary epidemic\n"
    "All countries vs. OECD-38  (entity & time FE; clustered SEs; 95% CI)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig("eipsa_oecd_vs_full_fd.png", dpi=200, bbox_inches="tight")
print(f"Full vs OECD comparison plot saved \u2192 eipsa_oecd_vs_full_fd.png")
plt.close()


# =====================================================================
# 7.  DIAGNOSTICS
# =====================================================================
print(f"\n\n{'=' * 76}")
print(" DIAGNOSTICS")
print(f"{'=' * 76}")

print(f"\n  Full panel:")
print(f"    Span:              {panel.index.get_level_values(1).min()}"
      f"\u2013{panel.index.get_level_values(1).max()}")
print(f"    Countries:         {panel.index.get_level_values(0).nunique()}")
print(f"    Epidemic c-yrs:    {int(panel['epidemic'].sum()):,}")

print(f"\n  OECD sub-panel:")
print(f"    Countries:         {n_oecd}")
print(f"    Epidemic c-yrs:    {n_epi_oecd}")
print(f"    Major epi c-yrs:   {n_major_oecd}")

print(f"\n  Outcome missingness (OECD):")
for v in OUTCOMES:
    nm = panel_oecd[v].isna().sum()
    print(f"    {v}: {nm:,} ({nm / len(panel_oecd) * 100:.1f}%)")
for v in OUTCOMES:
    dv = f"d_{v}"
    nm = panel_oecd[dv].isna().sum()
    print(f"    {dv}: {nm:,} ({nm / len(panel_oecd) * 100:.1f}%)")

print(textwrap.dedent("""
  IMPORTANT CAVEATS:
  1. This EM-DAT extract does NOT include COVID-19. Results reflect only
     non-COVID epidemics (SARS, MERS, H5N1, dengue, cholera, etc.).
  2. Only 24 epidemic records exist in OECD countries in this dataset,
     spanning 16 countries. Statistical power is very limited.
  3. 'Major epidemic' (deaths > 10) yields only 7 OECD country-years
     across 5 countries. Any results from this specification should be
     interpreted with extreme caution.
  4. First-differencing removes long-run trends but reduces sample size
     by one year per country and amplifies measurement noise.
"""))

print("Done.")
