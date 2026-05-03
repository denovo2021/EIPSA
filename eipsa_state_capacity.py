#!/usr/bin/env python3
"""
EIPSA – Final Figure: State Capacity & Selection
==================================================
"Less polarized societies adopted stricter lockdowns"
Pre-pandemic Social Polarization (2019) vs. Average Lockdown Stringency (2020-21)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

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

# ── Load OWID ──
owid = pd.read_csv(OWID_LOCAL, low_memory=False)
owid["date"] = pd.to_datetime(owid["date"])
owid = owid[~owid["iso_code"].str.startswith("OWID_", na=True)].copy()
owid["year"] = owid["date"].dt.year

owid_cy = (
    owid.groupby(["iso_code", "year"], as_index=False)
    .agg(stringency_mean=("stringency_index", "mean"))
    .rename(columns={"iso_code": "iso3"})
)

str_avg = (
    owid_cy[(owid_cy["iso3"].isin(OECD)) & (owid_cy["year"].isin([2020, 2021]))]
    .groupby("iso3")
    .agg(stringency_avg=("stringency_mean", "mean"))
    .reset_index()
)

# ── Load V-Dem (2019 only) ──
vdem = pd.read_csv(
    VDEM_PATH,
    usecols=["country_text_id", "country_name", "year", "v2smpolsoc"],
    low_memory=False,
)
vdem = vdem.rename(columns={"country_text_id": "iso3"})
vdem_2019 = vdem[vdem["year"] == 2019][["iso3", "v2smpolsoc"]].copy()

# ── Build cross-section ──
cs = pd.DataFrame({"iso3": OECD})
names = vdem[vdem["iso3"].isin(OECD)].drop_duplicates("iso3")[["iso3", "country_name"]]
cs = cs.merge(names, on="iso3", how="left")
cs = cs.merge(str_avg, on="iso3", how="left")
cs = cs.merge(vdem_2019, on="iso3", how="left")
cs = cs.dropna(subset=["v2smpolsoc", "stringency_avg"]).reset_index(drop=True)

print(f"N = {len(cs)} OECD countries with complete data\n")

# ── Regression ──
slope, intercept, r, p, se = sp_stats.linregress(
    cs["v2smpolsoc"], cs["stringency_avg"]
)
print(f"  r = {r:+.3f},  R2 = {r**2:.3f},  p = {p:.4f}")
print(f"  slope = {slope:+.3f},  intercept = {intercept:.1f}\n")

# ── Quadrant analysis ──
x_med = cs["v2smpolsoc"].median()
y_med = cs["stringency_avg"].median()

q_hi_coh = cs[(cs["v2smpolsoc"] < x_med) & (cs["stringency_avg"] >= y_med)]
q_hi_pol = cs[(cs["v2smpolsoc"] >= x_med) & (cs["stringency_avg"] < y_med)]
q_top_right = cs[(cs["v2smpolsoc"] >= x_med) & (cs["stringency_avg"] >= y_med)]
q_bot_left = cs[(cs["v2smpolsoc"] < x_med) & (cs["stringency_avg"] < y_med)]

print("  HIGH COHESION / HIGH STRINGENCY (top-left):")
for _, row in q_hi_coh.sort_values("stringency_avg", ascending=False).iterrows():
    print(f"    {row['iso3']}  ({row['country_name']})  "
          f"polsoc={row['v2smpolsoc']:+.2f}  stringency={row['stringency_avg']:.1f}")

print(f"\n  HIGH POLARIZATION / LOW STRINGENCY (bottom-right):")
for _, row in q_hi_pol.sort_values("v2smpolsoc", ascending=False).iterrows():
    print(f"    {row['iso3']}  ({row['country_name']})  "
          f"polsoc={row['v2smpolsoc']:+.2f}  stringency={row['stringency_avg']:.1f}")

# ── Plot ──
fig, ax = plt.subplots(figsize=(11, 8))

# Quadrant shading
ax.axvline(x_med, color="grey", linewidth=0.6, linestyle=":", alpha=0.5)
ax.axhline(y_med, color="grey", linewidth=0.6, linestyle=":", alpha=0.5)

ax.fill_between(
    [cs["v2smpolsoc"].min() - 0.3, x_med], y_med, 70,
    alpha=0.06, color="#2166ac", zorder=0,
)
ax.fill_between(
    [x_med, cs["v2smpolsoc"].max() + 0.3], 35, y_med,
    alpha=0.06, color="#b2182b", zorder=0,
)

# Quadrant labels
ax.text(
    cs["v2smpolsoc"].min() - 0.1, 66,
    "High Cohesion\nHigh Stringency",
    fontsize=8.5, fontstyle="italic", color="#2166ac", alpha=0.7,
    ha="left", va="top",
)
ax.text(
    cs["v2smpolsoc"].max() + 0.1, 39,
    "High Polarization\nLow Stringency",
    fontsize=8.5, fontstyle="italic", color="#b2182b", alpha=0.7,
    ha="right", va="bottom",
)

# Scatter points
ax.scatter(
    cs["v2smpolsoc"], cs["stringency_avg"],
    s=70, alpha=0.85, c="#4393c3", edgecolors="k", linewidths=0.6, zorder=4,
)

# Country labels with adjustText for overlap avoidance
from adjustText import adjust_text
texts = []
for _, row in cs.iterrows():
    t = ax.text(
        row["v2smpolsoc"], row["stringency_avg"],
        row["iso3"],
        fontsize=7.5, fontweight="bold", color="#333333",
        ha="center", va="center", zorder=5,
    )
    texts.append(t)
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#aaaaaa",
            lw=0.5), force_text=0.5, force_points=0.8)

# Regression line with CI band
xline = np.linspace(cs["v2smpolsoc"].min() - 0.2,
                     cs["v2smpolsoc"].max() + 0.2, 200)
yline = intercept + slope * xline

# Prediction interval
n = len(cs)
x_mean = cs["v2smpolsoc"].mean()
x_ss = ((cs["v2smpolsoc"] - x_mean) ** 2).sum()
resid_se = np.sqrt(
    ((cs["stringency_avg"] - intercept - slope * cs["v2smpolsoc"]) ** 2).sum()
    / (n - 2)
)
ci_half = 1.96 * resid_se * np.sqrt(1 / n + (xline - x_mean) ** 2 / x_ss)

ax.plot(xline, yline, color="#d6604d", linewidth=2.2, zorder=3)
ax.fill_between(xline, yline - ci_half, yline + ci_half,
                alpha=0.12, color="#d6604d", zorder=2)

# Stat annotation box
stat_text = (
    f"$r$ = {r:+.3f}    $R^2$ = {r**2:.3f}    $p$ = {p:.4f}\n"
    f"$N$ = {n} OECD countries"
)
ax.text(
    0.98, 0.03, stat_text,
    transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor="#999999", alpha=0.9),
)

# Axis labels
ax.set_xlabel("Pre-pandemic Social Polarization (2019)", fontsize=12.5)
ax.set_ylabel("Average Lockdown Stringency (2020\u20132021)", fontsize=12.5)
ax.set_title(
    "State Capacity as Selection:\n"
    "Socially Cohesive Countries Adopted Stricter Lockdowns",
    fontsize=14, fontweight="bold", pad=15,
)

ax.set_xlim(cs["v2smpolsoc"].min() - 0.4, cs["v2smpolsoc"].max() + 0.4)
ax.set_ylim(34, 68)
ax.tick_params(labelsize=10)

fig.tight_layout()
fig.savefig("eipsa_state_capacity.png", dpi=250, bbox_inches="tight")
print(f"\n  Plot saved -> eipsa_state_capacity.png")
plt.close()
print("Done.")
