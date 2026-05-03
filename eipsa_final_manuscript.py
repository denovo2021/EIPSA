#!/usr/bin/env python3
"""
EIPSA – Final Manuscript Assets
================================
Task 1: Figure 3 (fixed label overlaps with adjustText)
Task 2: Table 1 (publication-quality regression table)

Direction (confirmed):
  v2smpolsoc: HIGH = cohesive, LOW = polarized
  Outcome for regressions: polarization = -v2smpolsoc
  Positive coeff = increases polarization
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

# ── Paths & Constants ──
OWID_LOCAL = "./owid-covid-data.csv"
VDEM_PATH  = "./V-Dem-CY-FullOthers-v15_csv/V-Dem-CY-Full+Others-v15.csv"

OECD = sorted([
    "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE",
    "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "ISL",
    "IRL", "ISR", "ITA", "JPN", "KOR", "LVA", "LTU", "LUX",
    "MEX", "NLD", "NZL", "NOR", "POL", "PRT", "SVK", "SVN",
    "ESP", "SWE", "CHE", "TUR", "GBR", "USA",
])

def sig(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


# =================================================================
# DATA LOADING
# =================================================================
print("=" * 78)
print(" DATA LOADING")
print("=" * 78)

# ── OWID ──
owid = pd.read_csv(OWID_LOCAL, low_memory=False)
owid["date"] = pd.to_datetime(owid["date"])
owid = owid[~owid["iso_code"].str.startswith("OWID_", na=True)].copy()
owid["year"] = owid["date"].dt.year

owid_cy = (
    owid.groupby(["iso_code", "year"], as_index=False)
    .agg(
        covid_deaths_pm_eoy=("total_deaths_per_million", "max"),
        stringency_mean=("stringency_index", "mean"),
    )
    .rename(columns={"iso_code": "iso3"})
)
owid_cy["covid_deaths_pm_eoy"] = owid_cy["covid_deaths_pm_eoy"].fillna(0)
owid_cy = owid_cy.sort_values(["iso3", "year"])
owid_cy["covid_annual_deaths"] = (
    owid_cy.groupby("iso3")["covid_deaths_pm_eoy"]
    .diff().fillna(owid_cy["covid_deaths_pm_eoy"]).clip(lower=0)
)
owid_cy["covid_intensity"] = np.log1p(owid_cy["covid_annual_deaths"])
owid_cy["stringency_norm"] = owid_cy["stringency_mean"] / 100.0

# Mean stringency 2020-2021 (for cross-section)
str_avg = (
    owid_cy[(owid_cy["iso3"].isin(OECD)) & (owid_cy["year"].isin([2020, 2021]))]
    .groupby("iso3").agg(stringency_avg=("stringency_mean", "mean"))
    .reset_index()
)

# ── V-Dem ──
vdem = pd.read_csv(
    VDEM_PATH,
    usecols=["country_text_id", "country_name", "year", "v2smpolsoc"],
    low_memory=False,
)
vdem = vdem.rename(columns={"country_text_id": "iso3"})

# ── Panel for TWFE ──
panel = vdem[vdem["iso3"].isin(OECD)].merge(
    owid_cy[["iso3", "year", "covid_intensity", "stringency_norm"]],
    on=["iso3", "year"], how="left",
)
for c in ["covid_intensity", "stringency_norm"]:
    panel[c] = panel[c].fillna(0)
panel = panel.sort_values(["iso3", "year"])
panel["polarization"] = -panel["v2smpolsoc"]
panel["d_polarization"] = panel.groupby("iso3")["polarization"].diff()
panel = panel.set_index(["iso3", "year"])
panel_modern = panel.loc[panel.index.get_level_values("year") >= 2015].copy()

# ── Cross-section for Figure 3 ──
vdem_2019 = vdem[vdem["year"] == 2019][["iso3", "v2smpolsoc"]].copy()
names = vdem[vdem["iso3"].isin(OECD)].drop_duplicates("iso3")[["iso3", "country_name"]]
cs = pd.DataFrame({"iso3": OECD})
cs = cs.merge(names, on="iso3", how="left")
cs = cs.merge(str_avg, on="iso3", how="left")
cs = cs.merge(vdem_2019, on="iso3", how="left")
cs = cs.dropna(subset=["v2smpolsoc", "stringency_avg"]).reset_index(drop=True)

print(f"  Panel: {len(panel_modern)} obs, "
      f"{panel_modern.index.get_level_values(0).nunique()} countries")
print(f"  Cross-section: {len(cs)} countries")


# =================================================================
# TASK 1: FIGURE 3 — Fixed Label Overlaps
# =================================================================
print(f"\n{'=' * 78}")
print(" FIGURE 3: Selection Effect (fixed labels)")
print("=" * 78)

slope, intercept, r, p, se = sp_stats.linregress(
    cs["v2smpolsoc"], cs["stringency_avg"]
)
n = len(cs)
print(f"  r = {r:+.3f}, p = {p:.4f}, N = {n}")

x_med = cs["v2smpolsoc"].median()
y_med = cs["stringency_avg"].median()

sns.set_style("white")
sns.set_context("paper", font_scale=1.2)

fig3, ax3 = plt.subplots(figsize=(11, 8))
ax3.grid(True, linestyle="-", alpha=0.25, color="#cccccc", zorder=0)
ax3.set_axisbelow(True)

# Quadrant lines
ax3.axvline(x_med, color="#888888", linewidth=1.0, linestyle="--", alpha=0.5, zorder=1)
ax3.axhline(y_med, color="#888888", linewidth=1.0, linestyle="--", alpha=0.5, zorder=1)

# Quadrant shading
x_lo = cs["v2smpolsoc"].min() - 0.5
x_hi = cs["v2smpolsoc"].max() + 0.5
y_lo, y_hi = 33, 69

ax3.fill_between([x_lo, x_med], y_med, y_hi,
                 alpha=0.05, color="#b2182b", zorder=0)
ax3.fill_between([x_med, x_hi], y_lo, y_med,
                 alpha=0.05, color="#2166ac", zorder=0)

# Quadrant annotations
ax3.text(x_lo + 0.15, y_hi - 1.0,
         '"Coercive Response"\nLow Cohesion / High Stringency',
         fontsize=8.5, fontstyle="italic", color="#b2182b", alpha=0.6,
         ha="left", va="top")
ax3.text(x_hi - 0.15, y_lo + 1.0,
         '"Voluntary Compliance"\nHigh Cohesion / Low Stringency',
         fontsize=8.5, fontstyle="italic", color="#2166ac", alpha=0.6,
         ha="right", va="bottom")

# Regression line + 95% CI
xline = np.linspace(x_lo, x_hi, 300)
yline = intercept + slope * xline
x_mean = cs["v2smpolsoc"].mean()
x_ss = ((cs["v2smpolsoc"] - x_mean) ** 2).sum()
resid_var = ((cs["stringency_avg"] - intercept - slope * cs["v2smpolsoc"]) ** 2).sum() / (n - 2)
ci_half = sp_stats.t.ppf(0.975, n - 2) * np.sqrt(
    resid_var * (1.0 / n + (xline - x_mean) ** 2 / x_ss)
)
ax3.fill_between(xline, yline - ci_half, yline + ci_half,
                 alpha=0.12, color="#d6604d", zorder=2, label="95% CI")
ax3.plot(xline, yline, color="#d6604d", linewidth=2.0, zorder=3,
         label="OLS trend line")

# Scatter points — LARGER, distinct from labels
ax3.scatter(cs["v2smpolsoc"], cs["stringency_avg"],
            s=80, alpha=0.9, c="#4393c3", edgecolors="white", linewidths=1.0,
            zorder=5)

# Labels — offset from dots, then fine-tune with adjustText
texts = []
for _, row in cs.iterrows():
    # Place text initially offset ABOVE the point
    t = ax3.text(
        row["v2smpolsoc"], row["stringency_avg"] + 1.0,
        row["iso3"],
        fontsize=7.5, fontweight="bold", color="#222222",
        ha="center", va="bottom", zorder=6,
    )
    texts.append(t)

adjust_text(
    texts, ax=ax3,
    x=cs["v2smpolsoc"].values,
    y=cs["stringency_avg"].values,
    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5, shrinkA=0, shrinkB=5),
    force_text=(0.5, 0.5),
    force_points=(2.0, 2.0),
    expand_text=(1.1, 1.2),
    expand_points=(2.0, 2.0),
    min_arrow_len=5,
)

# Stat annotation (top-right)
stat_text = f"$r$ = {r:.3f}\n$p$ = {p:.4f}\n$N$ = {n}"
ax3.text(0.97, 0.97, stat_text,
         transform=ax3.transAxes, fontsize=11, ha="right", va="top",
         family="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                   edgecolor="#aaaaaa", alpha=0.95),
         zorder=7)

# Axes
ax3.set_xlabel(
    "Pre-pandemic Social Cohesion (2019)\n"
    "(V-Dem: v2smpolsoc;  low = polarized,  high = cohesive)",
    fontsize=12, labelpad=10,
)
ax3.set_ylabel("Mean Lockdown Stringency (2020\u20132021)", fontsize=12, labelpad=10)
ax3.set_title(
    "Figure 3.  The Selection Effect:\n"
    "Pre-pandemic Social Cohesion Determined Lockdown Strategy in OECD Countries",
    fontsize=13, fontweight="bold", pad=14,
)
ax3.set_xlim(x_lo, x_hi)
ax3.set_ylim(y_lo, y_hi)
ax3.tick_params(labelsize=10)

for spine in ax3.spines.values():
    spine.set_color("#cccccc")

fig3.tight_layout()
fig3.savefig("eipsa_figure3.png", dpi=300, bbox_inches="tight", facecolor="white")
print(f"  Saved -> eipsa_figure3.png (300 dpi)")
plt.close(fig3)


# =================================================================
# TASK 2: TABLE 1 — Publication-quality regression table
# =================================================================
print(f"\n{'=' * 78}")
print(" TABLE 1: TWFE First-Difference Regression Results")
print("=" * 78)

def get_coef(model, var):
    if model is None or var not in model.params.index:
        return None
    return {
        "b": model.params[var],
        "se": model.std_errors[var],
        "p": model.pvalues[var],
        "ci_lo": model.conf_int().loc[var, "lower"],
        "ci_hi": model.conf_int().loc[var, "upper"],
    }

def fmt_coef(c):
    if c is None:
        return "", ""
    return f"{c['b']:+.3f}{sig(c['p'])}", f"({c['se']:.3f})"

# Run models
m1 = PanelOLS(
    panel_modern["d_polarization"].dropna(),
    panel_modern.loc[panel_modern["d_polarization"].notna(), ["covid_intensity"]],
    entity_effects=True, time_effects=True,
).fit(cov_type="clustered", cluster_entity=True)

m2 = PanelOLS(
    panel_modern["d_polarization"].dropna(),
    panel_modern.loc[panel_modern["d_polarization"].notna(), ["stringency_norm"]],
    entity_effects=True, time_effects=True,
).fit(cov_type="clustered", cluster_entity=True)

cols3 = ["covid_intensity", "stringency_norm"]
valid3 = panel_modern[["d_polarization"] + cols3].dropna()
m3 = PanelOLS(
    valid3["d_polarization"], valid3[cols3],
    entity_effects=True, time_effects=True,
).fit(cov_type="clustered", cluster_entity=True)

# Model 4: Cross-sectional residualized
vdem_2024 = vdem[vdem["year"] == 2024][["iso3", "v2smpolsoc"]].rename(
    columns={"v2smpolsoc": "coh_2024"})
cs4 = cs.merge(vdem_2024, on="iso3", how="left").dropna().reset_index(drop=True)
cs4["d_pol"] = -(cs4["coh_2024"] - cs4["v2smpolsoc"])  # positive = more polarized
baseline = sm.add_constant(cs4["v2smpolsoc"])
resid_y = sm.OLS(cs4["d_pol"], baseline).fit().resid
resid_x = sm.OLS(cs4["stringency_avg"], baseline).fit().resid
m4 = sm.OLS(resid_y, sm.add_constant(resid_x)).fit(cov_type="HC1")
m4_b = m4.params.iloc[1]
m4_se = m4.bse.iloc[1]
m4_p = m4.pvalues.iloc[1]

# ── Format and print ──
W = 14  # column width

hdr1 = f"{'':>38s}  {'(1)':>{W}s}  {'(2)':>{W}s}  {'(3)':>{W}s}  {'(4)':>{W}s}"
hdr2 = f"{'':>38s}  {'Biological':>{W}s}  {'Policy':>{W}s}  {'Horse':>{W}s}  {'Selection':>{W}s}"
hdr3 = f"{'':>38s}  {'Threat':>{W}s}  {'Response':>{W}s}  {'Race':>{W}s}  {'(Residual.)':>{W}s}"
rule = "  " + "\u2500" * (38 + 4 * (W + 2))

print(f"\n  Dependent variable: {chr(916)} Social Polarization")
print(f"  (= {chr(8722)}{chr(916)}v2smpolsoc; positive = increased polarization)\n")
print(hdr1)
print(hdr2)
print(hdr3)
print(rule)

# COVID mortality row
c1 = get_coef(m1, "covid_intensity")
c3c = get_coef(m3, "covid_intensity")
b1, se1 = fmt_coef(c1)
b3c, se3c = fmt_coef(c3c)
print(f"  {'COVID-19 Mortality (log)':<38s}  {b1:>{W}s}  {'':>{W}s}  {b3c:>{W}s}  {'':>{W}s}")
print(f"  {'':38s}  {se1:>{W}s}  {'':>{W}s}  {se3c:>{W}s}  {'':>{W}s}")

# Stringency row
c2 = get_coef(m2, "stringency_norm")
c3s = get_coef(m3, "stringency_norm")
b2, se2 = fmt_coef(c2)
b3s, se3s = fmt_coef(c3s)
print(f"  {'Lockdown Stringency (0-1)':<38s}  {'':>{W}s}  {b2:>{W}s}  {b3s:>{W}s}  {'':>{W}s}")
print(f"  {'':38s}  {'':>{W}s}  {se2:>{W}s}  {se3s:>{W}s}  {'':>{W}s}")

# Model 4 row
b4_str = f"{m4_b:+.3f}{sig(m4_p)}"
se4_str = f"({m4_se:.3f})"
print(f"  {'Stringency (residualized)':<38s}  {'':>{W}s}  {'':>{W}s}  {'':>{W}s}  {b4_str:>{W}s}")
print(f"  {'':38s}  {'':>{W}s}  {'':>{W}s}  {'':>{W}s}  {se4_str:>{W}s}")

print(rule)

# Footer rows
def frow(label, v1, v2, v3, v4):
    print(f"  {label:<38s}  {v1:>{W}s}  {v2:>{W}s}  {v3:>{W}s}  {v4:>{W}s}")

frow("Observations", str(int(m1.nobs)), str(int(m2.nobs)),
     str(int(m3.nobs)), str(int(m4.nobs)))
frow("Countries", str(int(m1.entity_info["total"])),
     str(int(m2.entity_info["total"])),
     str(int(m3.entity_info["total"])), str(len(cs4)))
frow("R-squared (within)", f"{m1.rsquared_within:.4f}",
     f"{m2.rsquared_within:.4f}", f"{m3.rsquared_within:.4f}",
     f"{m4.rsquared:.4f}")
frow("Entity fixed effects", "Yes", "Yes", "Yes", "N/A")
frow("Year fixed effects", "Yes", "Yes", "Yes", "N/A")
frow("Standard errors", "Clustered", "Clustered", "Clustered", "HC1")

print(rule)
print(f"\n  Notes: Models 1\u20133 are two-way fixed-effects (TWFE) regressions on")
print(f"  first-differenced social polarization ({chr(916)}Polarization = "
      f"{chr(8722)}{chr(916)}v2smpolsoc).")
print(f"  Model 4 is cross-sectional OLS (N = {len(cs4)}) on the total 2019\u20132024")
print(f"  change, with both outcome and exposure residualized on 2019 baseline")
print(f"  cohesion (Frisch-Waugh-Lovell decomposition). Positive coefficients")
print(f"  indicate the exposure increases polarization.")
print(f"  * p < 0.10, ** p < 0.05, *** p < 0.01.")

# Print coefficient details for reference
print(f"\n  Model 3 details:")
for var, label in [("covid_intensity", "COVID Mortality"),
                   ("stringency_norm", "Stringency")]:
    c = get_coef(m3, var)
    print(f"    {label:>20s}: {chr(946)} = {c['b']:+.5f}, SE = {c['se']:.5f}, "
          f"95% CI [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}], p = {c['p']:.4f}")
print(f"    {'Resid. Stringency':>20s}: {chr(946)} = {m4_b:+.5f}, SE = {m4_se:.5f}, "
      f"p = {m4_p:.4f}")

print("\nDone.")
