#!/usr/bin/env python3
"""
EIPSA – Polarization Analysis  (v4)
=====================================
Tests whether COVID-19 mortality and lockdown stringency are associated
with rising political polarization in OECD countries.

Outcomes (V-Dem v15)
--------------------
- v2cacamps : Political polarization — "society is divided into
              antagonistic political camps."  Higher = MORE polarized.
- v2smpolsoc: Polarization of society (Digital Society module) —
              "political content on social media exacerbates political
              polarization."  Higher = MORE polarized.
- v2x_jucon : Judicial constraints on the executive (0–1 index).
              Lower = WEAKER constraints = more executive power.

Exposures (OWID COVID-19)
-------------------------
- covid_annual_intensity : ln(1 + annual COVID deaths per million)
- stringency_mean        : mean Oxford Stringency Index (0–100) per year

Requirements:
    pip install pandas openpyxl linearmodels matplotlib scipy
"""

from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import textwrap
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 0.  Paths & constants ──────────────────────────────────────────────
OWID_LOCAL = "./owid-covid-data.csv"
VDEM_PATH  = "./V-Dem-CY-FullOthers-v15_csv/V-Dem-CY-Full+Others-v15.csv"

OUTCOMES = {
    "v2cacamps":  "Political Polarization",
    "v2smpolsoc": "Social Polarization (digital)",
    "v2x_jucon":  "Judicial Constraints on Executive",
}

OECD = sorted([
    "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE",
    "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "ISL",
    "IRL", "ISR", "ITA", "JPN", "KOR", "LVA", "LTU", "LUX",
    "MEX", "NLD", "NZL", "NOR", "POL", "PRT", "SVK", "SVN",
    "ESP", "SWE", "CHE", "TUR", "GBR", "USA",
])

LAGS = [1, 2, 3]


# =====================================================================
# HELPERS
# =====================================================================
def sig(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "\u2020"
    return ""


def run_twfe(dep: str, exog: list[str], data: pd.DataFrame,
             min_obs: int = 30) -> dict | None:
    cols = [dep] + exog
    sub = data[cols].dropna()
    n_ent = sub.index.get_level_values(0).nunique()
    if sub.shape[0] < min_obs or n_ent < 3:
        return None
    mod = PanelOLS(sub[dep], sub[exog],
                   entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    t = exog[0]
    return {
        "beta":  res.params[t], "se": res.std_errors[t],
        "ci_lo": res.conf_int().loc[t, "lower"],
        "ci_hi": res.conf_int().loc[t, "upper"],
        "pval":  res.pvalues[t], "nobs": int(res.nobs),
        "n_ent": n_ent, "r2w": res.rsquared_within, "obj": res,
    }


def prow(label: str, r: dict | None) -> None:
    if r is None:
        print(f"  {label:>35s}:  insufficient obs"); return
    print(
        f"  {label:>35s}:  \u03b2={r['beta']:+.5f}  "
        f"SE={r['se']:.5f}  "
        f"CI[{r['ci_lo']:+.5f},{r['ci_hi']:+.5f}]  "
        f"p={r['pval']:.4f}{sig(r['pval'])}  "
        f"N={r['nobs']} n_c={r['n_ent']} R\u00b2w={r['r2w']:.4f}"
    )


# =====================================================================
# 1.  LOAD DATA
# =====================================================================
print("=" * 78)
print(" 1.  DATA LOADING")
print("=" * 78)

# ── OWID COVID ──
print("  OWID COVID-19 \u2026")
owid_raw = pd.read_csv(OWID_LOCAL, low_memory=False)
owid_raw["date"] = pd.to_datetime(owid_raw["date"])
owid = owid_raw[~owid_raw["iso_code"].str.startswith("OWID_", na=True)].copy()
owid["year"] = owid["date"].dt.year

owid_cy = (
    owid.groupby(["iso_code", "year"], as_index=False)
    .agg(
        covid_deaths_pm_eoy=("total_deaths_per_million", "max"),
        new_deaths_pm_sum=("new_deaths_per_million", "sum"),
        stringency_mean=("stringency_index", "mean"),
        stringency_max=("stringency_index", "max"),
    )
)
owid_cy = owid_cy.rename(columns={"iso_code": "iso3"})
owid_cy["covid_deaths_pm_eoy"] = owid_cy["covid_deaths_pm_eoy"].fillna(0)

# Annual incremental deaths/million
owid_cy = owid_cy.sort_values(["iso3", "year"])
owid_cy["covid_deaths_pm_annual"] = (
    owid_cy.groupby("iso3")["covid_deaths_pm_eoy"]
    .diff()
    .fillna(owid_cy["covid_deaths_pm_eoy"])
    .clip(lower=0)
)
owid_cy["covid_intensity"] = np.log1p(owid_cy["covid_deaths_pm_annual"])

# Stringency: normalise to 0-1 for coefficient interpretability
owid_cy["stringency_norm"] = owid_cy["stringency_mean"] / 100.0

print(f"  OWID: {len(owid_cy):,} country-years")

# ── V-Dem ──
print("  V-Dem v15 \u2026")
vdem_cols = ["country_text_id", "country_name", "year"] + list(OUTCOMES.keys())
vdem = pd.read_csv(VDEM_PATH, usecols=vdem_cols, low_memory=False)
vdem = vdem.rename(columns={"country_text_id": "iso3"})
print(f"  V-Dem: {len(vdem):,} rows")

# ── Merge ──
panel = vdem.merge(
    owid_cy[["iso3", "year", "covid_intensity",
             "covid_deaths_pm_annual", "covid_deaths_pm_eoy",
             "stringency_mean", "stringency_max", "stringency_norm"]],
    on=["iso3", "year"], how="left",
)
for c in ["covid_intensity", "covid_deaths_pm_annual",
          "covid_deaths_pm_eoy", "stringency_mean",
          "stringency_max", "stringency_norm"]:
    panel[c] = panel[c].fillna(0)

# Sort, first-diff, lags
panel = panel.sort_values(["iso3", "year"])
for v in OUTCOMES:
    panel[f"d_{v}"] = panel.groupby("iso3")[v].diff()

for k in LAGS:
    panel[f"covid_L{k}"] = panel.groupby("iso3")["covid_intensity"].shift(k)
    panel[f"string_L{k}"] = panel.groupby("iso3")["stringency_norm"].shift(k)

# Set index, OECD subsets
panel = panel.set_index(["iso3", "year"])
panel_oecd = panel.loc[panel.index.get_level_values(0).isin(OECD)].copy()
panel_oecd_modern = panel_oecd.loc[
    panel_oecd.index.get_level_values("year") >= 2000
].copy()

n_oecd = panel_oecd_modern.index.get_level_values(0).nunique()
print(f"\n  OECD panel (2000\u20132024): {len(panel_oecd_modern):,} obs, "
      f"{n_oecd} countries")


# =====================================================================
# 2.  DESCRIPTIVE TRENDS
# =====================================================================
print(f"\n{'=' * 78}")
print(" 2.  DESCRIPTIVE TRENDS  (OECD means, 2015\u20132024)")
print("=" * 78)

trends = (
    panel_oecd
    .loc[panel_oecd.index.get_level_values("year") >= 2015]
    .reset_index()
    .groupby("year")
    .agg(
        polarization=("v2cacamps", "mean"),
        soc_polar=("v2smpolsoc", "mean"),
        jucon=("v2x_jucon", "mean"),
        covid=("covid_intensity", "mean"),
        stringency=("stringency_norm", "mean"),
    )
)
print(trends.round(3).to_string())


# =====================================================================
# 3.  PANEL REGRESSIONS — COVID intensity
# =====================================================================
print(f"\n\n{'=' * 78}")
print(" 3.  TWFE REGRESSIONS \u2014 COVID intensity \u2192 polarization")
print("=" * 78)

exposure_covid = ["covid_intensity"] + [f"covid_L{k}" for k in LAGS]
exp_labels_c = {
    "covid_intensity": "t",
    **{f"covid_L{k}": f"t\u2013{k}" for k in LAGS},
}

# ── 3a. Level outcomes ──
print(f"\n{'─' * 78}")
print(" A. Levels: Y_it = \u03b1_i + \u03b3_t + \u03b2\u00b7COVID_it + \u03b5")
print(f"{'─' * 78}")
level_res = {}
for out, olabel in OUTCOMES.items():
    print(f"\n  {out}  ({olabel})")
    for ev in exposure_covid:
        lab = exp_labels_c[ev]
        r = run_twfe(out, [ev], panel_oecd_modern)
        level_res[(out, "covid", lab)] = r
        prow(f"COVID {lab}", r)

# ── 3b. First-differenced outcomes ──
print(f"\n{'─' * 78}")
print(" B. First-diff: \u0394Y_it = \u03b1_i + \u03b3_t + \u03b2\u00b7COVID_it + \u03b5")
print(f"{'─' * 78}")
fd_res = {}
for out, olabel in OUTCOMES.items():
    dv = f"d_{out}"
    print(f"\n  {dv}  (\u0394 {olabel})")
    for ev in exposure_covid:
        lab = exp_labels_c[ev]
        r = run_twfe(dv, [ev], panel_oecd_modern)
        fd_res[(dv, "covid", lab)] = r
        prow(f"COVID {lab}", r)
        if r and ev == "covid_intensity":
            print(r["obj"].summary)


# =====================================================================
# 4.  PANEL REGRESSIONS — Stringency (lockdown strictness)
# =====================================================================
print(f"\n\n{'=' * 78}")
print(" 4.  TWFE REGRESSIONS \u2014 Stringency \u2192 polarization")
print("=" * 78)
print("  Hypothesis: strict lockdowns caused polarization, independent")
print("  of the virus death toll itself.")

exposure_str = ["stringency_norm"] + [f"string_L{k}" for k in LAGS]
exp_labels_s = {
    "stringency_norm": "t",
    **{f"string_L{k}": f"t\u2013{k}" for k in LAGS},
}

# ── 4a. Levels ──
print(f"\n{'─' * 78}")
print(" A. Levels: Y_it = \u03b1_i + \u03b3_t + \u03b2\u00b7Stringency_it + \u03b5")
print(f"{'─' * 78}")
for out, olabel in OUTCOMES.items():
    print(f"\n  {out}  ({olabel})")
    for ev in exposure_str:
        lab = exp_labels_s[ev]
        r = run_twfe(out, [ev], panel_oecd_modern)
        level_res[(out, "string", lab)] = r
        prow(f"Stringency {lab}", r)

# ── 4b. First-diff ──
print(f"\n{'─' * 78}")
print(" B. First-diff: \u0394Y_it = \u03b1_i + \u03b3_t + \u03b2\u00b7Stringency_it + \u03b5")
print(f"{'─' * 78}")
for out, olabel in OUTCOMES.items():
    dv = f"d_{out}"
    print(f"\n  {dv}  (\u0394 {olabel})")
    for ev in exposure_str:
        lab = exp_labels_s[ev]
        r = run_twfe(dv, [ev], panel_oecd_modern)
        fd_res[(dv, "string", lab)] = r
        prow(f"Stringency {lab}", r)
        if r and ev == "stringency_norm":
            print(r["obj"].summary)


# =====================================================================
# 5.  HORSE RACE: COVID + Stringency together
# =====================================================================
print(f"\n\n{'=' * 78}")
print(" 5.  HORSE RACE \u2014 COVID + Stringency simultaneously")
print("=" * 78)
print("  Both exposures in the same regression to disentangle channels.")

for out, olabel in OUTCOMES.items():
    dv = f"d_{out}"
    print(f"\n  {dv}  (\u0394 {olabel})")
    r = run_twfe(dv, ["covid_intensity", "stringency_norm"],
                 panel_oecd_modern)
    if r:
        print(r["obj"].summary)
    else:
        print("    insufficient obs")


# =====================================================================
# 6.  CROSS-SECTIONAL SCATTER PLOTS
# =====================================================================
print(f"\n\n{'=' * 78}")
print(" 6.  CROSS-SECTIONAL ANALYSIS")
print("=" * 78)

vdem_flat = vdem[vdem["iso3"].isin(OECD)].copy()

cs = pd.DataFrame({"iso3": OECD})

# COVID cumulative deaths by end 2021
covid_cum = (
    owid_cy[(owid_cy["iso3"].isin(OECD)) & (owid_cy["year"] == 2021)]
    [["iso3", "covid_deaths_pm_eoy"]]
    .rename(columns={"covid_deaths_pm_eoy": "covid_pm_2021"})
)
cs = cs.merge(covid_cum, on="iso3", how="left")

# Max stringency (across all years)
str_max = (
    owid_cy[owid_cy["iso3"].isin(OECD)]
    .groupby("iso3")["stringency_max"]
    .max()
    .reset_index()
    .rename(columns={"stringency_max": "stringency_peak"})
)
# Mean stringency 2020-2021
str_mean = (
    owid_cy[(owid_cy["iso3"].isin(OECD)) & (owid_cy["year"].isin([2020, 2021]))]
    .groupby("iso3")["stringency_mean"]
    .mean()
    .reset_index()
    .rename(columns={"stringency_mean": "stringency_avg_20_21"})
)
cs = cs.merge(str_max, on="iso3", how="left")
cs = cs.merge(str_mean, on="iso3", how="left")

# Outcome levels at key years
for yr in [2019, 2022, 2023, 2024]:
    for v in OUTCOMES:
        yr_data = vdem_flat[vdem_flat["year"] == yr][["iso3", v]].rename(
            columns={v: f"{v}_{yr}"}
        )
        cs = cs.merge(yr_data, on="iso3", how="left")

# Changes
for v in OUTCOMES:
    cs[f"d_{v}_19_22"] = cs[f"{v}_2022"] - cs[f"{v}_2019"]
    cs[f"d_{v}_19_24"] = cs[f"{v}_2024"] - cs[f"{v}_2019"]

cs["ln_covid"] = np.log1p(cs["covid_pm_2021"].fillna(0))

# Country names
names = vdem_flat.drop_duplicates("iso3")[["iso3", "country_name"]]
cs = cs.merge(names, on="iso3", how="left")

# Print cross-section
print("\n  Change in polarization (2019 \u2192 2024), sorted by COVID deaths:")
print(cs[["iso3", "covid_pm_2021", "stringency_avg_20_21",
          "d_v2cacamps_19_24", "d_v2smpolsoc_19_24", "d_v2x_jucon_19_24"]]
      .sort_values("covid_pm_2021", ascending=False)
      .round(3).to_string(index=False))

# ── OLS regressions ──
print("\n  Cross-sectional OLS:")
scatter_specs = [
    ("ln_covid",             "ln(COVID deaths/M)",
     "d_v2cacamps_19_24",    "\u0394 Political Polarization 2019\u21922024"),
    ("ln_covid",             "ln(COVID deaths/M)",
     "d_v2smpolsoc_19_24",   "\u0394 Social Polarization 2019\u21922024"),
    ("stringency_avg_20_21", "Mean Stringency 2020\u201321",
     "d_v2cacamps_19_24",    "\u0394 Political Polarization 2019\u21922024"),
    ("stringency_avg_20_21", "Mean Stringency 2020\u201321",
     "d_v2smpolsoc_19_24",   "\u0394 Social Polarization 2019\u21922024"),
]

for xvar, xlabel, yvar, ylabel in scatter_specs:
    valid = cs.dropna(subset=[xvar, yvar])
    slope, intercept, r, p, se = sp_stats.linregress(valid[xvar], valid[yvar])
    print(f"    {xlabel} \u2192 {ylabel}:")
    print(f"      slope={slope:+.4f}  R\u00b2={r**2:.3f}  p={p:.4f}{sig(p)}")

# ── 4-panel scatter ──
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
highlight = ["USA", "GBR", "HUN", "TUR", "POL", "MEX", "NZL",
             "AUS", "JPN", "ITA", "ESP", "KOR", "ISR", "CZE",
             "SVN", "COL", "GRC", "FRA", "DEU", "SWE"]

for ax, (xvar, xlabel, yvar, ylabel) in zip(axes.flat, scatter_specs):
    valid = cs.dropna(subset=[xvar, yvar])
    ax.scatter(valid[xvar], valid[yvar], s=50, alpha=0.7,
               edgecolors="k", linewidths=0.5, zorder=3)
    for _, row in valid.iterrows():
        if row["iso3"] in highlight:
            ax.annotate(row["iso3"], (row[xvar], row[yvar]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(4, 3), textcoords="offset points")
    slope, intercept, r, p, se = sp_stats.linregress(valid[xvar], valid[yvar])
    xline = np.linspace(valid[xvar].min(), valid[xvar].max(), 100)
    ax.plot(xline, intercept + slope * xline, "r--", linewidth=1.5, alpha=0.8)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}\nslope={slope:+.4f}, R\u00b2={r**2:.3f}, "
                 f"p={p:.4f}", fontsize=10)

fig.suptitle(
    "OECD countries: COVID exposure & lockdown strictness\n"
    "vs. change in polarization (V-Dem, 2019 \u2192 2024)",
    fontsize=13, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("eipsa_polarization_scatter.png", dpi=200, bbox_inches="tight")
print(f"\n  Scatter plot saved \u2192 eipsa_polarization_scatter.png")
plt.close()


# =====================================================================
# 7.  DESCRIPTIVE TIMELINE PLOT
# =====================================================================
trends_full = (
    panel_oecd
    .loc[panel_oecd.index.get_level_values("year") >= 2000]
    .reset_index()
    .groupby("year")
    .agg(
        camps_mean=("v2cacamps", "mean"),
        camps_sd=("v2cacamps", "std"),
        polsoc_mean=("v2smpolsoc", "mean"),
        polsoc_sd=("v2smpolsoc", "std"),
        jucon_mean=("v2x_jucon", "mean"),
        covid=("covid_intensity", "mean"),
        string=("stringency_norm", "mean"),
    )
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
color1, color2 = "#1f77b4", "#d62728"

# Panel 1: Political polarization
ax = axes[0]
ax.fill_between(trends_full.index,
                trends_full["camps_mean"] - trends_full["camps_sd"],
                trends_full["camps_mean"] + trends_full["camps_sd"],
                alpha=0.15, color=color1)
ax.plot(trends_full.index, trends_full["camps_mean"],
        "o-", color=color1, linewidth=2, markersize=4, label="v2cacamps")
ax.set_ylabel("Political Polarization (v2cacamps)", color=color1)
ax.tick_params(axis="y", labelcolor=color1)
ax2 = ax.twinx()
ax2.bar(trends_full.index, trends_full["covid"],
        color=color2, alpha=0.3, width=0.7, label="COVID intensity")
ax2.set_ylabel("COVID intensity", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)
ax.axvspan(2019.5, 2024.5, alpha=0.08, color="red")
ax.set_title("Political Polarization", fontweight="bold")
ax.set_xlabel("Year")

# Panel 2: Social polarization
ax = axes[1]
ax.fill_between(trends_full.index,
                trends_full["polsoc_mean"] - trends_full["polsoc_sd"],
                trends_full["polsoc_mean"] + trends_full["polsoc_sd"],
                alpha=0.15, color=color1)
ax.plot(trends_full.index, trends_full["polsoc_mean"],
        "o-", color=color1, linewidth=2, markersize=4, label="v2smpolsoc")
ax.set_ylabel("Social Polarization (v2smpolsoc)", color=color1)
ax.tick_params(axis="y", labelcolor=color1)
ax2 = ax.twinx()
ax2.bar(trends_full.index, trends_full["string"],
        color="#ff7f0e", alpha=0.3, width=0.7, label="Stringency")
ax2.set_ylabel("Stringency (0\u20131)", color="#ff7f0e")
ax2.tick_params(axis="y", labelcolor="#ff7f0e")
ax.axvspan(2019.5, 2024.5, alpha=0.08, color="red")
ax.set_title("Social Polarization", fontweight="bold")
ax.set_xlabel("Year")

# Panel 3: Judicial constraints
ax = axes[2]
ax.fill_between(trends_full.index,
                trends_full["jucon_mean"] - 0,  # no sd for 0-1 index
                trends_full["jucon_mean"],
                alpha=0.15, color=color1)
ax.plot(trends_full.index, trends_full["jucon_mean"],
        "o-", color=color1, linewidth=2, markersize=4, label="v2x_jucon")
ax.set_ylabel("Judicial Constraints (v2x_jucon)", color=color1)
ax.tick_params(axis="y", labelcolor=color1)
ax2 = ax.twinx()
ax2.bar(trends_full.index, trends_full["covid"],
        color=color2, alpha=0.3, width=0.7, label="COVID intensity")
ax2.set_ylabel("COVID intensity", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)
ax.axvspan(2019.5, 2024.5, alpha=0.08, color="red")
ax.set_title("Judicial Constraints", fontweight="bold")
ax.set_xlabel("Year")

fig.suptitle(
    "OECD means: polarization & judicial constraints vs. COVID/stringency\n"
    "(V-Dem; shaded band = \u00b11 SD across countries)",
    fontsize=12, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig("eipsa_polarization_timeline.png", dpi=200, bbox_inches="tight")
print(f"  Timeline saved \u2192 eipsa_polarization_timeline.png")
plt.close()


# =====================================================================
# 8.  SUMMARY TABLE
# =====================================================================
print(f"\n\n{'=' * 78}")
print(" 8.  RESULTS SUMMARY")
print("=" * 78)

print(textwrap.dedent("""
  OECD MEAN POLARIZATION TRENDS (v2cacamps, higher = more polarized):
    2019: -0.581    2020: -0.497    2021: -0.240
    2022: -0.224    2023: -0.175    2024: -0.069
    Total shift 2019-2024: +0.512  (massive increase in polarization)
"""))

print("  TWFE First-Differenced Results  (contemporaneous, OECD 2000-2024):")
print(f"  {'Outcome':<35s} {'Exposure':<20s} {'beta':>8s} {'SE':>8s} "
      f"{'p':>8s} {'sig':>4s}")
print("  " + "\u2500" * 83)
for out in OUTCOMES:
    dv = f"d_{out}"
    dlabel = f"\u0394 {OUTCOMES[out]}"
    for exp_type, exp_label in [("covid", "COVID intensity"),
                                 ("string", "Stringency")]:
        key = (dv, exp_type, "t")
        r = fd_res.get(key)
        if r:
            print(f"  {dlabel:<35s} {exp_label:<20s} "
                  f"{r['beta']:>+8.5f} {r['se']:>8.5f} "
                  f"{r['pval']:>8.4f} {sig(r['pval']):>4s}")
        else:
            print(f"  {dlabel:<35s} {exp_label:<20s} {'N/A':>8s}")

print(textwrap.dedent("""
  INTERPRETATION GUIDE:
    v2cacamps  : Higher = MORE polarized.  Beta > 0 means exposure
                 is associated with INCREASING polarization.
    v2smpolsoc : Higher = MORE polarized (social media dimension).
    v2x_jucon  : Higher = STRONGER judicial constraints.
                 Beta < 0 means exposure weakens constraints.

  KEY CAVEATS:
    1. Polarization was already rising pre-COVID (see timeline).
       Year FE partially absorb this, but a common acceleration
       in 2020+ would be captured by time FE, not by the exposure.
    2. Stringency and COVID deaths are correlated — the horse-race
       regression (Section 5) helps disentangle them.
    3. N = 38 OECD countries over 25 years; within-country variation
       in stringency is limited (all countries locked down ~simultaneously).
"""))

print("Done.")
