#!/usr/bin/env python3
"""
EIPSA — 01: Main Analysis (Table 1 & Figures 1–2)
====================================================
Reproduces all core results from:
  "The Stress Test That Changed Nothing"

Outputs:
  - Table 1: TWFE first-difference regressions (Models 1–4)
  - Figure 1: OECD social cohesion trend (2010–2024)
  - Figure 2: Coefficient plot (horse-race model)

Direction (confirmed empirically):
  v2smpolsoc: HIGH = cohesive, LOW = polarized
  Outcome:    polarization = −v2smpolsoc
              positive coefficient = exposure INCREASES polarization
"""

import os
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

# ── Paths (relative to this script) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(SCRIPT_DIR, os.pardir)
DATA_PATH  = os.path.join(ROOT_DIR, "data", "EIPSA_OECD_panel_2019_2024.csv")
FIG_DIR    = os.path.join(ROOT_DIR, "output", "figures")
TBL_DIR    = os.path.join(ROOT_DIR, "output", "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)


def sig(p):
    """Significance stars."""
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


# =================================================================
# 1. DATA LOADING & PREPARATION
# =================================================================
print("=" * 78)
print(" EIPSA — 01: Main Analysis")
print("=" * 78)

df = pd.read_csv(DATA_PATH)
print(f"  Loaded: {len(df)} rows, {df['iso3'].nunique()} countries, "
      f"years {df['year'].min()}\u2013{df['year'].max()}")

# ── Build panel (2015–2024 for TWFE) ──
panel = df[df["year"] >= 2015].copy()
panel = panel.sort_values(["iso3", "year"])

# Outcome: polarization = −v2smpolsoc (positive = more polarized)
panel["polarization"] = -panel["v2smpolsoc"]
panel["d_polarization"] = panel.groupby("iso3")["polarization"].diff()

panel = panel.set_index(["iso3", "year"])

# ── Build cross-section (for Model 4 and selection effect) ──
row_2019 = df[df["year"] == 2019][["iso3", "v2smpolsoc"]].rename(
    columns={"v2smpolsoc": "cohesion_2019"})
row_2024 = df[df["year"] == 2024][["iso3", "v2smpolsoc"]].rename(
    columns={"v2smpolsoc": "cohesion_2024"})

cs = (
    df[["iso3", "country_name", "stringency_avg_2020_2021",
        "gdp_per_capita_2019"]].drop_duplicates("iso3")
    .merge(row_2019, on="iso3", how="left")
    .merge(row_2024, on="iso3", how="left")
    .dropna(subset=["cohesion_2019", "stringency_avg_2020_2021"])
    .reset_index(drop=True)
)
cs["d_pol"] = -(cs["cohesion_2024"] - cs["cohesion_2019"])  # positive = more polarized

print(f"  Panel: {len(panel)} obs | Cross-section: {len(cs)} countries")


# =================================================================
# 2. TABLE 1 — TWFE Regressions
# =================================================================
print(f"\n{'=' * 78}")
print(" TABLE 1: TWFE First-Difference Regression Results")
print("=" * 78)

# Helper to extract coefficients
def get_coef(model, var):
    if model is None or var not in model.params.index:
        return None
    return {
        "b": model.params[var], "se": model.std_errors[var],
        "p": model.pvalues[var],
        "ci_lo": model.conf_int().loc[var, "lower"],
        "ci_hi": model.conf_int().loc[var, "upper"],
    }

def fmt_coef(c):
    if c is None:
        return "", ""
    return f"{c['b']:+.3f}{sig(c['p'])}", f"({c['se']:.3f})"

# ── Model 1: Biological Threat ──
m1 = PanelOLS(
    panel["d_polarization"].dropna(),
    panel.loc[panel["d_polarization"].notna(), ["covid_intensity"]],
    entity_effects=True, time_effects=True,
).fit(cov_type="clustered", cluster_entity=True)

# ── Model 2: Policy Response ──
m2 = PanelOLS(
    panel["d_polarization"].dropna(),
    panel.loc[panel["d_polarization"].notna(), ["stringency_norm"]],
    entity_effects=True, time_effects=True,
).fit(cov_type="clustered", cluster_entity=True)

# ── Model 3: Horse Race ──
cols3 = ["covid_intensity", "stringency_norm"]
valid3 = panel[["d_polarization"] + cols3].dropna()
m3 = PanelOLS(
    valid3["d_polarization"], valid3[cols3],
    entity_effects=True, time_effects=True,
).fit(cov_type="clustered", cluster_entity=True)

# ── Model 4: Residualized (FWL decomposition) ──
cs4 = cs.dropna(subset=["d_pol"]).reset_index(drop=True)
baseline = sm.add_constant(cs4["cohesion_2019"])
resid_y = sm.OLS(cs4["d_pol"], baseline).fit().resid
resid_x = sm.OLS(cs4["stringency_avg_2020_2021"], baseline).fit().resid
m4 = sm.OLS(resid_y, sm.add_constant(resid_x)).fit(cov_type="HC1")
m4_b, m4_se, m4_p = m4.params.iloc[1], m4.bse.iloc[1], m4.pvalues.iloc[1]

# ── Print table ──
W = 14
tbl_lines = []

def tprint(line=""):
    print(line)
    tbl_lines.append(line)

tprint(f"  Dependent variable: \u0394 Social Polarization")
tprint(f"  (= \u2212\u0394v2smpolsoc; positive = increased polarization)\n")

hdr1 = f"{'':>38s}  {'(1)':>{W}s}  {'(2)':>{W}s}  {'(3)':>{W}s}  {'(4)':>{W}s}"
hdr2 = f"{'':>38s}  {'Biological':>{W}s}  {'Policy':>{W}s}  {'Horse':>{W}s}  {'Selection':>{W}s}"
hdr3 = f"{'':>38s}  {'Threat':>{W}s}  {'Response':>{W}s}  {'Race':>{W}s}  {'(Residual.)':>{W}s}"
rule = "  " + "\u2500" * (38 + 4 * (W + 2))

for line in [hdr1, hdr2, hdr3, rule]:
    tprint(line)

# Mortality
c1   = get_coef(m1, "covid_intensity")
c3c  = get_coef(m3, "covid_intensity")
b1, se1   = fmt_coef(c1)
b3c, se3c = fmt_coef(c3c)
tprint(f"  {'COVID-19 Mortality (log)':<38s}  {b1:>{W}s}  {'':>{W}s}  {b3c:>{W}s}  {'':>{W}s}")
tprint(f"  {'':38s}  {se1:>{W}s}  {'':>{W}s}  {se3c:>{W}s}  {'':>{W}s}")

# Stringency
c2   = get_coef(m2, "stringency_norm")
c3s  = get_coef(m3, "stringency_norm")
b2, se2   = fmt_coef(c2)
b3s, se3s = fmt_coef(c3s)
lbl_str = "Lockdown Stringency (0\u20131)"
tprint(f"  {lbl_str:<38s}  {'':>{W}s}  {b2:>{W}s}  {b3s:>{W}s}  {'':>{W}s}")
tprint(f"  {'':38s}  {'':>{W}s}  {se2:>{W}s}  {se3s:>{W}s}  {'':>{W}s}")

# Model 4
b4_str = f"{m4_b:+.3f}{sig(m4_p)}"
se4_str = f"({m4_se:.3f})"
tprint(f"  {'Stringency (residualized)':<38s}  {'':>{W}s}  {'':>{W}s}  {'':>{W}s}  {b4_str:>{W}s}")
tprint(f"  {'':38s}  {'':>{W}s}  {'':>{W}s}  {'':>{W}s}  {se4_str:>{W}s}")

tprint(rule)

def frow(label, v1, v2, v3, v4):
    tprint(f"  {label:<38s}  {v1:>{W}s}  {v2:>{W}s}  {v3:>{W}s}  {v4:>{W}s}")

frow("Observations", str(int(m1.nobs)), str(int(m2.nobs)),
     str(int(m3.nobs)), str(int(m4.nobs)))
frow("Countries", str(int(m1.entity_info["total"])),
     str(int(m2.entity_info["total"])),
     str(int(m3.entity_info["total"])), str(len(cs4)))
frow("R\u00b2 (within)", f"{m1.rsquared_within:.4f}",
     f"{m2.rsquared_within:.4f}", f"{m3.rsquared_within:.4f}",
     f"{m4.rsquared:.4f}")
frow("Entity fixed effects", "Yes", "Yes", "Yes", "N/A")
frow("Year fixed effects", "Yes", "Yes", "Yes", "N/A")
frow("Standard errors", "Clustered", "Clustered", "Clustered", "HC1")

tprint(rule)
tprint(f"\n  Notes: Models 1\u20133 are TWFE on \u0394Polarization = \u2212\u0394v2smpolsoc.")
tprint(f"  Model 4: Cross-sectional OLS (N = {len(cs4)}), "
       f"Frisch-Waugh-Lovell decomposition.")
tprint(f"  * p < 0.10, ** p < 0.05, *** p < 0.01.")

# Save table
tbl_path = os.path.join(TBL_DIR, "table1.txt")
with open(tbl_path, "w") as f:
    f.write("\n".join(tbl_lines))
print(f"\n  Saved -> {tbl_path}")


# =================================================================
# 3. FIGURE 1 — OECD Social Cohesion Trend (2010–2024)
# =================================================================
print(f"\n{'=' * 78}")
print(" FIGURE 1: OECD Social Cohesion Trend")
print("=" * 78)

sns.set_style("white")
sns.set_context("paper", font_scale=1.2)

trend = (
    df[(df["year"] >= 2010) & (df["year"] <= 2024)]
    .groupby("year")
    .agg(mean_coh=("v2smpolsoc", "mean"),
         se_coh=("v2smpolsoc", "sem"))
    .reset_index()
)
trend["ci95"] = 1.96 * trend["se_coh"]

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.fill_between(trend["year"],
                 trend["mean_coh"] - trend["ci95"],
                 trend["mean_coh"] + trend["ci95"],
                 alpha=0.2, color="#4393c3")
ax1.plot(trend["year"], trend["mean_coh"], "o-", color="#2166ac",
         linewidth=2.2, markersize=7, markerfacecolor="white",
         markeredgewidth=2, zorder=3)

ax1.axvline(2020, color="#d6604d", linewidth=1.5, linestyle="--", alpha=0.7)
ax1.text(2020.1, trend["mean_coh"].max() + 0.05, "COVID-19\nonset",
         fontsize=9, color="#d6604d", va="bottom")

for yr in [2010, 2019, 2024]:
    row = trend[trend["year"] == yr].iloc[0]
    ax1.annotate(f"{row['mean_coh']:+.2f}",
                 (yr, row["mean_coh"]),
                 textcoords="offset points", xytext=(0, 12),
                 ha="center", fontsize=9, fontweight="bold")

ax1.set_xlabel("Year", fontsize=12)
ax1.set_ylabel("Mean Social Cohesion (v2smpolsoc)\nacross 38 OECD Countries",
               fontsize=12)
ax1.set_title("Figure 1.  The Structural Decline:\n"
              "OECD Social Cohesion Has Been Falling Since 2010",
              fontsize=13, fontweight="bold", pad=14)
ax1.set_xlim(2009.5, 2024.5)
ax1.grid(axis="y", alpha=0.3)
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)

fig1.tight_layout()
fig1_path = os.path.join(FIG_DIR, "figure1.png")
fig1.savefig(fig1_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"  Saved -> {fig1_path}")
plt.close(fig1)


# =================================================================
# 4. FIGURE 2 — Coefficient Plot (Horse-Race Model)
# =================================================================
print(f"\n{'=' * 78}")
print(" FIGURE 2: Coefficient Plot")
print("=" * 78)

labels = ["COVID-19 Mortality\n(log deaths/million)",
          "Lockdown Stringency\n(0\u20131 index)"]
betas  = [m3.params["covid_intensity"], m3.params["stringency_norm"]]
ci_lo  = [m3.conf_int().loc["covid_intensity", "lower"],
          m3.conf_int().loc["stringency_norm", "lower"]]
ci_hi  = [m3.conf_int().loc["covid_intensity", "upper"],
          m3.conf_int().loc["stringency_norm", "upper"]]
pvals  = [m3.pvalues["covid_intensity"], m3.pvalues["stringency_norm"]]

fig2, ax2 = plt.subplots(figsize=(8, 4.5))
colors = ["#4393c3", "#d6604d"]
y_pos  = [1, 0]

for i in range(2):
    ax2.errorbar(betas[i], y_pos[i],
                 xerr=[[betas[i] - ci_lo[i]], [ci_hi[i] - betas[i]]],
                 fmt="o", color=colors[i], markersize=10, capsize=6,
                 capthick=2, elinewidth=2, markeredgecolor="white",
                 markeredgewidth=1, zorder=4)
    p_str = f"p = {pvals[i]:.3f}" if pvals[i] >= 0.001 else "p < 0.001"
    ax2.text(ci_hi[i] + 0.05, y_pos[i], p_str, fontsize=10,
             va="center", color=colors[i])

ax2.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels, fontsize=11)
ax2.set_xlabel("Coefficient Estimate\n"
               "(\u0394 Polarization = \u2212\u0394v2smpolsoc; "
               "positive = more polarized)", fontsize=11)
ax2.set_title("Figure 2.  Horse-Race Model: Neither Mortality nor Stringency\n"
              "Predicts Changes in Social Polarization",
              fontsize=12, fontweight="bold", pad=14)
ax2.grid(axis="x", alpha=0.3)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)

fig2.tight_layout()
fig2_path = os.path.join(FIG_DIR, "figure2.png")
fig2.savefig(fig2_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"  Saved -> {fig2_path}")
plt.close(fig2)


# =================================================================
# 5. SUMMARY
# =================================================================
print(f"\n{'=' * 78}")
print(" Summary of Key Results")
print("=" * 78)
print(f"  Model 1 (Mortality):   b = {c1['b']:+.4f}, p = {c1['p']:.4f}")
print(f"  Model 2 (Stringency):  b = {c2['b']:+.4f}, p = {c2['p']:.4f}")
print(f"  Model 3 (Horse Race):  Mortality b = {c3c['b']:+.4f} (p={c3c['p']:.4f}), "
      f"Stringency b = {c3s['b']:+.4f} (p={c3s['p']:.4f})")
print(f"  Model 4 (Residual.):   b = {m4_b:+.4f}, p = {m4_p:.4f}")
print(f"\n  Selection effect:      r = {sp_stats.pearsonr(cs['cohesion_2019'], cs['stringency_avg_2020_2021'])[0]:.3f}, "
      f"p = {sp_stats.pearsonr(cs['cohesion_2019'], cs['stringency_avg_2020_2021'])[1]:.4f}")
print("\n  Done.")
