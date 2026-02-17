#!/usr/bin/env python3
"""
EIPSA — 02: The Selection Effect (Figure 3)
=============================================
Reproduces Figure 3 from:
  "The Stress Test That Changed Nothing"

Figure 3: Pre-pandemic social cohesion (v2smpolsoc, 2019) vs. mean lockdown
stringency (2020–2021) across 38 OECD countries.  r = −0.51, p = 0.001.

Output:
  - output/figures/figure3.png  (300 dpi, publication-ready)
"""

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

# ── Paths (relative to this script) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(SCRIPT_DIR, os.pardir)
DATA_PATH  = os.path.join(ROOT_DIR, "data", "EIPSA_OECD_panel_2019_2024.csv")
FIG_DIR    = os.path.join(ROOT_DIR, "output", "figures")

os.makedirs(FIG_DIR, exist_ok=True)


# =================================================================
# 1. DATA PREPARATION
# =================================================================
print("=" * 78)
print(" EIPSA — 02: Selection Effect (Figure 3)")
print("=" * 78)

df = pd.read_csv(DATA_PATH)

# Build 38-country cross-section
cs = (
    df[["iso3", "country_name", "stringency_avg_2020_2021"]]
    .drop_duplicates("iso3")
    .copy()
)
coh_2019 = df[df["year"] == 2019][["iso3", "v2smpolsoc"]].rename(
    columns={"v2smpolsoc": "cohesion_2019"})
cs = cs.merge(coh_2019, on="iso3", how="left")
cs = cs.dropna(subset=["cohesion_2019", "stringency_avg_2020_2021"])
cs = cs.reset_index(drop=True)

# Regression
slope, intercept, r, p, se = sp_stats.linregress(
    cs["cohesion_2019"], cs["stringency_avg_2020_2021"])
n = len(cs)

print(f"  N = {n} OECD countries")
print(f"  r = {r:.3f}, p = {p:.4f}")
print(f"  slope = {slope:+.3f}, intercept = {intercept:.1f}")


# =================================================================
# 2. QUADRANT ANALYSIS
# =================================================================
x_med = cs["cohesion_2019"].median()
y_med = cs["stringency_avg_2020_2021"].median()

vol  = cs[(cs["cohesion_2019"] >= x_med) &
          (cs["stringency_avg_2020_2021"] < y_med)]
coer = cs[(cs["cohesion_2019"] < x_med) &
          (cs["stringency_avg_2020_2021"] >= y_med)]

print(f"\n  Medians: cohesion = {x_med:+.2f}, stringency = {y_med:.1f}")
print(f"  'Voluntary Compliance' ({len(vol)}): "
      f"{', '.join(vol.sort_values('cohesion_2019', ascending=False)['iso3'])}")
print(f"  'Coercive Response'    ({len(coer)}): "
      f"{', '.join(coer.sort_values('stringency_avg_2020_2021', ascending=False)['iso3'])}")


# =================================================================
# 3. FIGURE 3
# =================================================================
print(f"\n  Generating Figure 3...")

sns.set_style("white")
sns.set_context("paper", font_scale=1.2)

fig, ax = plt.subplots(figsize=(11, 8))
ax.grid(True, linestyle="-", alpha=0.25, color="#cccccc", zorder=0)
ax.set_axisbelow(True)

# Axis limits
x_lo = cs["cohesion_2019"].min() - 0.5
x_hi = cs["cohesion_2019"].max() + 0.5
y_lo, y_hi = 33, 69

# Quadrant lines
ax.axvline(x_med, color="#888888", linewidth=1.0, linestyle="--",
           alpha=0.5, zorder=1)
ax.axhline(y_med, color="#888888", linewidth=1.0, linestyle="--",
           alpha=0.5, zorder=1)

# Quadrant shading
ax.fill_between([x_lo, x_med], y_med, y_hi,
                alpha=0.05, color="#b2182b", zorder=0)
ax.fill_between([x_med, x_hi], y_lo, y_med,
                alpha=0.05, color="#2166ac", zorder=0)

# Quadrant annotations
ax.text(x_lo + 0.15, y_hi - 1.0,
        "\u201cCoercive Response\u201d\nLow Cohesion / High Stringency",
        fontsize=8.5, fontstyle="italic", color="#b2182b", alpha=0.6,
        ha="left", va="top")
ax.text(x_hi - 0.15, y_lo + 1.0,
        "\u201cVoluntary Compliance\u201d\nHigh Cohesion / Low Stringency",
        fontsize=8.5, fontstyle="italic", color="#2166ac", alpha=0.6,
        ha="right", va="bottom")

# Regression line + 95 % confidence band
xline = np.linspace(x_lo, x_hi, 300)
yline = intercept + slope * xline
x_mean = cs["cohesion_2019"].mean()
x_ss   = ((cs["cohesion_2019"] - x_mean) ** 2).sum()
resid_var = ((cs["stringency_avg_2020_2021"]
              - intercept - slope * cs["cohesion_2019"]) ** 2).sum() / (n - 2)
ci_half = sp_stats.t.ppf(0.975, n - 2) * np.sqrt(
    resid_var * (1.0 / n + (xline - x_mean) ** 2 / x_ss))

ax.fill_between(xline, yline - ci_half, yline + ci_half,
                alpha=0.12, color="#d6604d", zorder=2, label="95 % CI")
ax.plot(xline, yline, color="#d6604d", linewidth=2.0, zorder=3,
        label="OLS trend line")

# Scatter points
ax.scatter(cs["cohesion_2019"], cs["stringency_avg_2020_2021"],
           s=80, alpha=0.9, c="#4393c3", edgecolors="white",
           linewidths=1.0, zorder=5)

# Country labels (with overlap avoidance)
texts = []
for _, row in cs.iterrows():
    t = ax.text(row["cohesion_2019"],
                row["stringency_avg_2020_2021"] + 1.0,
                row["iso3"],
                fontsize=7.5, fontweight="bold", color="#222222",
                ha="center", va="bottom", zorder=6)
    texts.append(t)

adjust_text(
    texts, ax=ax,
    x=cs["cohesion_2019"].values,
    y=cs["stringency_avg_2020_2021"].values,
    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5,
                    shrinkA=0, shrinkB=5),
    force_text=(0.5, 0.5),
    force_points=(2.0, 2.0),
    expand_text=(1.1, 1.2),
    expand_points=(2.0, 2.0),
    min_arrow_len=5,
)

# Stat annotation box
stat_text = f"$r$ = {r:.3f}\n$p$ = {p:.4f}\n$N$ = {n}"
ax.text(0.97, 0.97, stat_text,
        transform=ax.transAxes, fontsize=11, ha="right", va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#aaaaaa", alpha=0.95),
        zorder=7)

# Labels & title
ax.set_xlabel(
    "Pre-pandemic Social Cohesion (2019)\n"
    "(V-Dem: v2smpolsoc;  low = polarized,  high = cohesive)",
    fontsize=12, labelpad=10)
ax.set_ylabel("Mean Lockdown Stringency (2020\u20132021)",
              fontsize=12, labelpad=10)
ax.set_title(
    "Figure 3.  The Selection Effect:\n"
    "Pre-pandemic Social Cohesion Determined Lockdown Strategy "
    "in OECD Countries",
    fontsize=13, fontweight="bold", pad=14)
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)
ax.tick_params(labelsize=10)

for spine in ax.spines.values():
    spine.set_color("#cccccc")

fig.tight_layout()
fig3_path = os.path.join(FIG_DIR, "figure3.png")
fig.savefig(fig3_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"  Saved -> {fig3_path}")
plt.close(fig)

print("\n  Done.")
