#!/usr/bin/env python3
"""
EIPSA – Figure 3: The Paradox of Pandemic Response
====================================================
Pre-pandemic Social Cohesion vs. Lockdown Stringency (OECD, N=38)
Publication-ready scatter with quadrant analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

# ── Paths ──
OWID_LOCAL = "./owid-covid-data.csv"
VDEM_PATH  = "./V-Dem-CY-FullOthers-v15_csv/V-Dem-CY-Full+Others-v15.csv"

OECD = sorted([
    "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE",
    "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "ISL",
    "IRL", "ISR", "ITA", "JPN", "KOR", "LVA", "LTU", "LUX",
    "MEX", "NLD", "NZL", "NOR", "POL", "PRT", "SVK", "SVN",
    "ESP", "SWE", "CHE", "TUR", "GBR", "USA",
])


# =================================================================
# 1. DATA PREPARATION
# =================================================================
print("=" * 70)
print(" EIPSA Figure 3 — Data Preparation")
print("=" * 70)

# ── OWID: Mean Stringency 2020-2021 ──
owid = pd.read_csv(OWID_LOCAL, low_memory=False)
owid["date"] = pd.to_datetime(owid["date"])
owid = owid[~owid["iso_code"].str.startswith("OWID_", na=True)].copy()
owid["year"] = owid["date"].dt.year

str_avg = (
    owid[(owid["iso_code"].isin(OECD)) & (owid["year"].isin([2020, 2021]))]
    .groupby("iso_code")
    .agg(stringency_avg=("stringency_index", "mean"))
    .reset_index()
    .rename(columns={"iso_code": "iso3"})
)

# ── V-Dem: Social Polarization at 2019 ──
vdem = pd.read_csv(
    VDEM_PATH,
    usecols=["country_text_id", "country_name", "year", "v2smpolsoc"],
    low_memory=False,
)
vdem = vdem.rename(columns={"country_text_id": "iso3"})
vdem_2019 = vdem[vdem["year"] == 2019][["iso3", "v2smpolsoc"]].copy()

# ── Merge ──
cs = pd.DataFrame({"iso3": OECD})
names = vdem[vdem["iso3"].isin(OECD)].drop_duplicates("iso3")[["iso3", "country_name"]]
cs = cs.merge(names, on="iso3", how="left")
cs = cs.merge(str_avg, on="iso3", how="left")
cs = cs.merge(vdem_2019, on="iso3", how="left")
cs = cs.dropna(subset=["v2smpolsoc", "stringency_avg"]).reset_index(drop=True)

# Rescale v2smpolsoc to a 0-4 interpretive range for the axis label
# V-Dem v2smpolsoc is standardised ~N(0,1), range roughly -4 to +4 in OECD
# We keep raw values but note the direction in the axis label.

print(f"  Sample: N = {len(cs)} OECD countries")
print(f"  v2smpolsoc range: [{cs['v2smpolsoc'].min():.2f}, {cs['v2smpolsoc'].max():.2f}]")
print(f"  Stringency range: [{cs['stringency_avg'].min():.1f}, {cs['stringency_avg'].max():.1f}]")

# ── Regression ──
slope, intercept, r, p, se = sp_stats.linregress(
    cs["v2smpolsoc"], cs["stringency_avg"]
)
n = len(cs)
print(f"\n  Pearson r = {r:+.3f}")
print(f"  R-squared = {r**2:.3f}")
print(f"  p-value   = {p:.4f}")
print(f"  slope     = {slope:+.3f}")


# =================================================================
# 2. FIGURE 3
# =================================================================
print(f"\n{'=' * 70}")
print(" Generating Figure 3...")
print("=" * 70)

# ── Style ──
sns.set_style("white")
sns.set_context("paper", font_scale=1.2)

fig, ax = plt.subplots(figsize=(10, 7.5))

# Light grid
ax.grid(True, linestyle="-", alpha=0.3, color="#cccccc", zorder=0)
ax.set_axisbelow(True)

# ── Quadrant medians ──
x_med = cs["v2smpolsoc"].median()
y_med = cs["stringency_avg"].median()

ax.axvline(x_med, color="#888888", linewidth=1.0, linestyle="--", alpha=0.6,
           zorder=1)
ax.axhline(y_med, color="#888888", linewidth=1.0, linestyle="--", alpha=0.6,
           zorder=1)

# ── Quadrant labels ──
x_min = cs["v2smpolsoc"].min() - 0.5
x_max = cs["v2smpolsoc"].max() + 0.5
y_min = 33
y_max = 69

ax.text(
    x_min + 0.15, y_max - 1.2,
    "\"Coercive Response\"",
    fontsize=9, fontstyle="italic", color="#b2182b", alpha=0.65,
    ha="left", va="top", zorder=1,
)
ax.text(
    x_max - 0.15, y_min + 1.0,
    "\"Voluntary Compliance\"",
    fontsize=9, fontstyle="italic", color="#2166ac", alpha=0.65,
    ha="right", va="bottom", zorder=1,
)

# ── Regression line + 95% CI ──
xline = np.linspace(x_min, x_max, 300)
yline = intercept + slope * xline

x_mean = cs["v2smpolsoc"].mean()
x_ss = ((cs["v2smpolsoc"] - x_mean) ** 2).sum()
resid_var = ((cs["stringency_avg"] - intercept - slope * cs["v2smpolsoc"]) ** 2).sum() / (n - 2)
ci_half = sp_stats.t.ppf(0.975, n - 2) * np.sqrt(
    resid_var * (1.0 / n + (xline - x_mean) ** 2 / x_ss)
)

ax.fill_between(xline, yline - ci_half, yline + ci_half,
                alpha=0.15, color="#d6604d", zorder=2, label="95% CI")
ax.plot(xline, yline, color="#d6604d", linewidth=2.0, zorder=3,
        label="OLS trend line")

# ── Scatter points ──
ax.scatter(
    cs["v2smpolsoc"], cs["stringency_avg"],
    s=55, alpha=0.90, c="#4393c3", edgecolors="white", linewidths=0.8,
    zorder=4,
)

# ── ISO labels (adjustText) ──
texts = []
for _, row in cs.iterrows():
    t = ax.text(
        row["v2smpolsoc"], row["stringency_avg"],
        row["iso3"],
        fontsize=7.5, fontweight="bold", color="#222222",
        ha="center", va="center", zorder=5,
    )
    texts.append(t)

adjust_text(
    texts, ax=ax,
    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.4),
    force_text=(0.6, 0.6),
    force_points=(0.8, 0.8),
    expand_text=(1.15, 1.25),
    expand_points=(1.3, 1.3),
)

# ── Stat box (top-right) ──
stat_text = f"$r$ = {r:.3f}\n$p$ = {p:.4f}\n$N$ = {n}"
ax.text(
    0.97, 0.97, stat_text,
    transform=ax.transAxes, fontsize=11, ha="right", va="top",
    family="monospace",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
              edgecolor="#aaaaaa", alpha=0.95),
    zorder=6,
)

# ── Axes ──
ax.set_xlabel(
    "Pre-pandemic Social Cohesion (2019)\n"
    "(V-Dem: v2smpolsoc;  low = polarized,  high = cohesive)",
    fontsize=12, labelpad=10,
)
ax.set_ylabel(
    "Mean Lockdown Stringency (2020\u20132021)",
    fontsize=12, labelpad=10,
)
ax.set_title(
    "Figure 3.  The Paradox of Pandemic Response:\n"
    "Social Cohesion Predicted Lockdown Stringency Across OECD Countries",
    fontsize=13, fontweight="bold", pad=14,
)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.tick_params(labelsize=11)

# Spine styling
for spine in ax.spines.values():
    spine.set_color("#cccccc")

fig.tight_layout()
fig.savefig("eipsa_figure3.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print(f"  Saved -> eipsa_figure3.png  (300 dpi)")
plt.close()


# =================================================================
# 3. QUADRANT ANALYSIS
# =================================================================
print(f"\n{'=' * 70}")
print(" Quadrant Analysis")
print(f"  Median social cohesion (v2smpolsoc): {x_med:+.2f}")
print(f"  Median stringency:                   {y_med:.1f}")
print("=" * 70)

# Voluntary Compliance: high cohesion (x > median), low stringency (y < median)
vol = cs[(cs["v2smpolsoc"] >= x_med) & (cs["stringency_avg"] < y_med)]
vol = vol.sort_values("v2smpolsoc", ascending=False)

# Coercive Response: low cohesion (x < median), high stringency (y >= median)
coer = cs[(cs["v2smpolsoc"] < x_med) & (cs["stringency_avg"] >= y_med)]
coer = coer.sort_values("stringency_avg", ascending=False)

# Off-diagonal: high cohesion + high stringency
hh = cs[(cs["v2smpolsoc"] >= x_med) & (cs["stringency_avg"] >= y_med)]
hh = hh.sort_values("stringency_avg", ascending=False)

# Off-diagonal: low cohesion + low stringency
ll = cs[(cs["v2smpolsoc"] < x_med) & (cs["stringency_avg"] < y_med)]
ll = ll.sort_values("v2smpolsoc")

print(f"\n  VOLUNTARY COMPLIANCE (High Cohesion / Low Stringency)  [{len(vol)} countries]")
for _, row in vol.iterrows():
    print(f"    {row['iso3']:>3s}  {row['country_name']:<30s}  "
          f"cohesion={row['v2smpolsoc']:+.2f}  stringency={row['stringency_avg']:.1f}")

print(f"\n  COERCIVE RESPONSE (Low Cohesion / High Stringency)  [{len(coer)} countries]")
for _, row in coer.iterrows():
    print(f"    {row['iso3']:>3s}  {row['country_name']:<30s}  "
          f"cohesion={row['v2smpolsoc']:+.2f}  stringency={row['stringency_avg']:.1f}")

print(f"\n  HIGH COHESION + HIGH STRINGENCY (off-diagonal)  [{len(hh)} countries]")
for _, row in hh.iterrows():
    print(f"    {row['iso3']:>3s}  {row['country_name']:<30s}  "
          f"cohesion={row['v2smpolsoc']:+.2f}  stringency={row['stringency_avg']:.1f}")

print(f"\n  LOW COHESION + LOW STRINGENCY (off-diagonal)  [{len(ll)} countries]")
for _, row in ll.iterrows():
    print(f"    {row['iso3']:>3s}  {row['country_name']:<30s}  "
          f"cohesion={row['v2smpolsoc']:+.2f}  stringency={row['stringency_avg']:.1f}")

print("\nDone.")
