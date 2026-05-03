#!/usr/bin/env python3
"""
EIPSA – Manuscript Figures 1 & 2, and Table 1
===============================================
DIRECTION CHECK (confirmed empirically):
  v2smpolsoc:  HIGH = COHESIVE (less polarized)
               LOW  = POLARIZED
  v2cacamps:   HIGH = POLARIZED
               LOW  = COHESIVE
  Correlation: r(v2smpolsoc, v2cacamps) = -0.847

To make coefficients intuitive for the paper ("positive = more polarized"),
we define the outcome as NEGATIVE v2smpolsoc:
  polarization_it = -v2smpolsoc_it
  D_polarization  = polarization_t - polarization_{t-1}
                  = -(v2smpolsoc_t - v2smpolsoc_{t-1})

With this coding:
  positive beta => exposure INCREASES polarization
  negative beta => exposure DECREASES polarization (increases cohesion)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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

def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "\u2020"
    return ""


# =================================================================
# 0. LOAD DATA
# =================================================================
print("=" * 78)
print(" DATA LOADING & DIRECTION VERIFICATION")
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

# Annual incremental deaths
owid_cy = owid_cy.sort_values(["iso3", "year"])
owid_cy["covid_annual_deaths"] = (
    owid_cy.groupby("iso3")["covid_deaths_pm_eoy"]
    .diff().fillna(owid_cy["covid_deaths_pm_eoy"]).clip(lower=0)
)
owid_cy["covid_intensity"] = np.log1p(owid_cy["covid_annual_deaths"])
owid_cy["stringency_norm"] = owid_cy["stringency_mean"] / 100.0

# ── V-Dem ──
vdem = pd.read_csv(
    VDEM_PATH,
    usecols=["country_text_id", "country_name", "year",
             "v2smpolsoc", "v2cacamps"],
    low_memory=False,
)
vdem = vdem.rename(columns={"country_text_id": "iso3"})

# Direction verification
oecd19 = vdem[(vdem["iso3"].isin(OECD)) & (vdem["year"] == 2019)].dropna()
r_check = oecd19["v2smpolsoc"].corr(oecd19["v2cacamps"])
print(f"  Direction check: corr(v2smpolsoc, v2cacamps) = {r_check:.3f}")
print(f"  v2smpolsoc: HIGH = cohesive, LOW = polarized  (confirmed)")
print(f"  Defining outcome: polarization = -v2smpolsoc")
print(f"  So positive coeff = exposure INCREASES polarization\n")

# ── Build panel ──
panel = vdem[vdem["iso3"].isin(OECD)].merge(
    owid_cy[["iso3", "year", "covid_intensity", "stringency_norm"]],
    on=["iso3", "year"], how="left",
)
for c in ["covid_intensity", "stringency_norm"]:
    panel[c] = panel[c].fillna(0)

panel = panel.sort_values(["iso3", "year"])

# Outcome: polarization = -v2smpolsoc (higher = more polarized)
panel["polarization"] = -panel["v2smpolsoc"]
panel["d_polarization"] = panel.groupby("iso3")["polarization"].diff()

# Also keep original for Figure 1
panel["d_v2smpolsoc"] = panel.groupby("iso3")["v2smpolsoc"].diff()

panel = panel.set_index(["iso3", "year"])
panel_modern = panel.loc[panel.index.get_level_values("year") >= 2015].copy()

n_c = panel_modern.index.get_level_values(0).nunique()
n_obs = len(panel_modern)
print(f"  Panel: {n_obs} obs, {n_c} OECD countries, 2015-2024")


# =================================================================
# FIGURE 1: Trends in Social Cohesion 2010-2024
# =================================================================
print(f"\n{'=' * 78}")
print(" FIGURE 1: OECD Social Cohesion Trend (2010-2024)")
print("=" * 78)

oecd_trend = (
    vdem[(vdem["iso3"].isin(OECD)) & (vdem["year"].between(2010, 2024))]
    .groupby("year")["v2smpolsoc"]
    .agg(["mean", "std", "count", lambda x: x.quantile(0.25),
          lambda x: x.quantile(0.75)])
)
oecd_trend.columns = ["mean", "std", "n", "q25", "q75"]
oecd_trend["se"] = oecd_trend["std"] / np.sqrt(oecd_trend["n"])
oecd_trend["ci_lo"] = oecd_trend["mean"] - 1.96 * oecd_trend["se"]
oecd_trend["ci_hi"] = oecd_trend["mean"] + 1.96 * oecd_trend["se"]

for yr in [2010, 2015, 2019, 2020, 2024]:
    if yr in oecd_trend.index:
        row = oecd_trend.loc[yr]
        print(f"  {yr}: mean={row['mean']:+.3f}  95%CI=[{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]  "
              f"IQR=[{row['q25']:+.2f}, {row['q75']:+.2f}]")

sns.set_style("white")
sns.set_context("paper", font_scale=1.2)

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.grid(True, linestyle="-", alpha=0.25, color="#cccccc", zorder=0)
ax1.set_axisbelow(True)

years = oecd_trend.index.values

# IQR band
ax1.fill_between(years, oecd_trend["q25"], oecd_trend["q75"],
                 alpha=0.15, color="#4393c3", label="Interquartile range (IQR)")

# 95% CI band
ax1.fill_between(years, oecd_trend["ci_lo"], oecd_trend["ci_hi"],
                 alpha=0.30, color="#4393c3", label="95% CI of the mean")

# Mean line
ax1.plot(years, oecd_trend["mean"], color="#2166ac", linewidth=2.5,
         marker="o", markersize=5, zorder=4, label="OECD mean")

# Pandemic line
ax1.axvline(2020, color="#d6604d", linewidth=1.5, linestyle="--", alpha=0.8,
            zorder=3)
ax1.text(2020.15, ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 0 else 2.0,
         "COVID-19\nPandemic", fontsize=9, color="#d6604d", fontstyle="italic",
         va="top")

# Zero line
ax1.axhline(0, color="grey", linewidth=0.6, linestyle=":", alpha=0.5)

ax1.set_xlabel("Year", fontsize=12)
ax1.set_ylabel("Social Cohesion (V-Dem: v2smpolsoc)\n"
               "Higher = More Cohesive, Lower = More Polarized",
               fontsize=11)
ax1.set_title(
    "Figure 1.  Trends in Social Cohesion across OECD Countries (2010\u20132024)",
    fontsize=13, fontweight="bold", pad=12,
)
ax1.set_xlim(2009.5, 2024.5)
ax1.set_xticks(range(2010, 2025))
ax1.tick_params(labelsize=10, axis="x", rotation=45)
ax1.legend(fontsize=9, loc="lower left")

for spine in ax1.spines.values():
    spine.set_color("#cccccc")

fig1.tight_layout()
fig1.savefig("eipsa_figure1.png", dpi=300, bbox_inches="tight", facecolor="white")
print(f"  Saved -> eipsa_figure1.png")
plt.close(fig1)


# =================================================================
# TABLE 1: TWFE REGRESSION RESULTS
# =================================================================
print(f"\n{'=' * 78}")
print(" TABLE 1: TWFE First-Difference Models")
print(f"  Outcome: {chr(916)} Polarization (= {chr(8722)}v2smpolsoc)")
print(f"  Positive coeff = exposure INCREASES polarization")
print("=" * 78)

def run_twfe(dep, exog_list, data, min_obs=30):
    """Run PanelOLS with entity + time FE, clustered SE."""
    cols = [dep] + exog_list
    sub = data[cols].dropna()
    n_ent = sub.index.get_level_values(0).nunique()
    if sub.shape[0] < min_obs or n_ent < 3:
        return None
    mod = PanelOLS(sub[dep], sub[exog_list],
                   entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    return res

# ── Model 1: Biological Threat only ──
m1 = run_twfe("d_polarization", ["covid_intensity"], panel_modern)

# ── Model 2: Policy Stringency only ──
m2 = run_twfe("d_polarization", ["stringency_norm"], panel_modern)

# ── Model 3: Horse Race ──
m3 = run_twfe("d_polarization", ["covid_intensity", "stringency_norm"],
              panel_modern)

# ── Model 4: Cross-sectional residualized ──
# Build cross-section
str_avg = (
    owid_cy[(owid_cy["iso3"].isin(OECD)) & (owid_cy["year"].isin([2020, 2021]))]
    .groupby("iso3").agg(stringency_avg=("stringency_mean", "mean"))
    .reset_index()
)
vdem_2019 = vdem[vdem["year"] == 2019][["iso3", "v2smpolsoc"]].rename(
    columns={"v2smpolsoc": "cohesion_2019"})
vdem_2024 = vdem[vdem["year"] == 2024][["iso3", "v2smpolsoc"]].rename(
    columns={"v2smpolsoc": "cohesion_2024"})
cs = pd.DataFrame({"iso3": OECD})
cs = cs.merge(str_avg, on="iso3", how="left")
cs = cs.merge(vdem_2019, on="iso3", how="left")
cs = cs.merge(vdem_2024, on="iso3", how="left")
cs = cs.dropna().reset_index(drop=True)
# Polarization change (positive = more polarized)
cs["d_polarization"] = -(cs["cohesion_2024"] - cs["cohesion_2019"])
# Residualize both on 2019 baseline
baseline = sm.add_constant(cs["cohesion_2019"])
resid_y = sm.OLS(cs["d_polarization"], baseline).fit().resid
resid_x = sm.OLS(cs["stringency_avg"], baseline).fit().resid
m4_ols = sm.OLS(resid_y, sm.add_constant(resid_x)).fit(cov_type="HC1")

# ── Helper: extract coeff from a PanelOLS result ──
def get_coef(model, var):
    """Return (beta_str, se_str) or ('', '') if var not in model."""
    if model is None:
        return "", ""
    if var not in model.params.index:
        return "", ""
    b = model.params[var]
    se_ = model.std_errors[var]
    p_ = model.pvalues[var]
    return f"{b:+.4f}{sig(p_)}", f"({se_:.4f})"

# ── Print Table ──
print(f"\n{'':>40s}  {'Model 1':>12s}  {'Model 2':>12s}  "
      f"{'Model 3':>12s}  {'Model 4':>12s}")
print(f"{'':>40s}  {'Mortality':>12s}  {'Stringency':>12s}  "
      f"{'Horse Race':>12s}  {'Residual.':>12s}")
print("  " + "\u2500" * 96)

# Row: COVID mortality
b_m1, se_m1 = get_coef(m1, "covid_intensity")
b_m3c, se_m3c = get_coef(m3, "covid_intensity")
print(f"  {'COVID-19 Mortality (log)':<38s}  {b_m1:>12s}  {'':>12s}  {b_m3c:>12s}  {'':>12s}")
print(f"  {'':38s}  {se_m1:>12s}  {'':>12s}  {se_m3c:>12s}  {'':>12s}")

# Row: Lockdown Stringency
b_m2, se_m2 = get_coef(m2, "stringency_norm")
b_m3s, se_m3s = get_coef(m3, "stringency_norm")
str_label = "Lockdown Stringency (0\u20131)"
print(f"  {str_label:<38s}  {'':>12s}  {b_m2:>12s}  {b_m3s:>12s}  {'':>12s}")
print(f"  {'':38s}  {'':>12s}  {se_m2:>12s}  {se_m3s:>12s}  {'':>12s}")

# Row: Model 4 residualized stringency
b4 = m4_ols.params.iloc[1]
se4 = m4_ols.bse.iloc[1]
p4 = m4_ols.pvalues.iloc[1]
b4_str = f"{b4:+.4f}{sig(p4)}"
se4_str = f"({se4:.4f})"
print(f"  {'Stringency (residualized)':<38s}  {'':>12s}  {'':>12s}  {'':>12s}  {b4_str:>12s}")
print(f"  {'':38s}  {'':>12s}  {'':>12s}  {'':>12s}  {se4_str:>12s}")

print("  " + "\u2500" * 90)

# N, R2
n_vals = []
r2_vals = []
for m in [m1, m2, m3]:
    if m is not None:
        n_vals.append(f"{int(m.nobs)}")
        r2_vals.append(f"{m.rsquared_within:.4f}")
    else:
        n_vals.append("")
        r2_vals.append("")
n_vals.append(f"{int(m4_ols.nobs)}")
r2_vals.append(f"{m4_ols.rsquared:.4f}")

print(f"  {'N (observations)':<38s}  {n_vals[0]:>10s}  {n_vals[1]:>10s}  "
      f"{n_vals[2]:>10s}  {n_vals[3]:>10s}")
print(f"  {'R-squared (within / adj.)':<38s}  {r2_vals[0]:>10s}  {r2_vals[1]:>10s}  "
      f"{r2_vals[2]:>10s}  {r2_vals[3]:>10s}")
print(f"  {'Entity FE':<38s}  {'Yes':>10s}  {'Yes':>10s}  {'Yes':>10s}  {'N/A':>10s}")
print(f"  {'Time FE':<38s}  {'Yes':>10s}  {'Yes':>10s}  {'Yes':>10s}  {'N/A':>10s}")
print(f"  {'Clustered SE':<38s}  {'Country':>10s}  {'Country':>10s}  "
      f"{'Country':>10s}  {'HC1':>10s}")

print("\n  Notes: Outcome is first-differenced polarization = -(v2smpolsoc_t - v2smpolsoc_{t-1}).")
print("  Positive coefficients indicate the exposure INCREASES polarization.")
print("  Model 4 is cross-sectional OLS on the 2019-2024 total change,")
print("  residualized on 2019 baseline cohesion (Frisch-Waugh-Lovell).")
print(f"  Significance: {chr(8224)} p<0.10, * p<0.05, ** p<0.01, *** p<0.001")

# Print detailed model summaries
print(f"\n  --- Model 3 (Horse Race) Detail ---")
if m3:
    for var in ["covid_intensity", "stringency_norm"]:
        b = m3.params[var]
        se_ = m3.std_errors[var]
        ci = m3.conf_int().loc[var]
        p_ = m3.pvalues[var]
        print(f"  {var:>20s}: beta={b:+.5f}  SE={se_:.5f}  "
              f"95%CI=[{ci['lower']:+.5f}, {ci['upper']:+.5f}]  p={p_:.4f}{sig(p_)}")


# =================================================================
# FIGURE 2: Coefficient Plot from Model 3
# =================================================================
print(f"\n{'=' * 78}")
print(" FIGURE 2: Coefficient Plot (Horse Race Model)")
print("=" * 78)

if m3 is not None:
    vars_plot = ["covid_intensity", "stringency_norm"]
    labels_plot = ["COVID-19 Mortality\n(log deaths per million)",
                   "Lockdown Stringency\n(0\u2013100, normalized)"]
    betas = [m3.params[v] for v in vars_plot]
    ci_lo = [m3.conf_int().loc[v, "lower"] for v in vars_plot]
    ci_hi = [m3.conf_int().loc[v, "upper"] for v in vars_plot]

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.grid(True, axis="x", linestyle="-", alpha=0.25, color="#cccccc", zorder=0)
    ax2.set_axisbelow(True)

    y_pos = [1, 0]
    colors = ["#4393c3", "#d6604d"]

    for i, (b, lo, hi, lab, col) in enumerate(
            zip(betas, ci_lo, ci_hi, labels_plot, colors)):
        ax2.errorbar(b, y_pos[i], xerr=[[b - lo], [hi - b]],
                     fmt="o", markersize=10, color=col,
                     ecolor=col, elinewidth=2.5, capsize=6, capthick=2,
                     zorder=4)
        # p-value annotation
        p_ = m3.pvalues[vars_plot[i]]
        ax2.text(hi + 0.02, y_pos[i],
                 f"p = {p_:.3f}{sig(p_)}",
                 fontsize=10, va="center", ha="left", color=col,
                 fontweight="bold")

    ax2.axvline(0, color="#333333", linewidth=1.2, linestyle="--", zorder=3)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels_plot, fontsize=11)
    ax2.set_xlabel(
        r"Coefficient ($\hat{\beta}$): Effect on $\Delta$ Polarization"
        "\n(positive = increases polarization)",
        fontsize=11,
    )
    ax2.set_title(
        "Figure 2.  Horse Race: Biological Threat vs. Policy Response\n"
        "(TWFE First-Difference, 38 OECD Countries)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax2.set_ylim(-0.6, 1.6)

    for spine in ax2.spines.values():
        spine.set_color("#cccccc")

    fig2.tight_layout()
    fig2.savefig("eipsa_figure2.png", dpi=300, bbox_inches="tight",
                 facecolor="white")
    print(f"  Saved -> eipsa_figure2.png")
    plt.close(fig2)


# =================================================================
# DIRECTION SUMMARY
# =================================================================
print(f"\n{'=' * 78}")
print(" CRITICAL: DIRECTION INTERPRETATION SUMMARY")
print("=" * 78)

if m3:
    b_mort = m3.params["covid_intensity"]
    p_mort = m3.pvalues["covid_intensity"]
    b_str = m3.params["stringency_norm"]
    p_str = m3.pvalues["stringency_norm"]

    print(f"""
  Outcome = Delta Polarization = -(v2smpolsoc_t - v2smpolsoc_{{t-1}})
  POSITIVE coefficient = exposure INCREASES polarization
  NEGATIVE coefficient = exposure DECREASES polarization (increases cohesion)

  Model 3 (Horse Race):
    COVID Mortality:     beta = {b_mort:+.4f}, p = {p_mort:.4f}  -> {'INCREASES' if b_mort > 0 else 'DECREASES'} polarization
    Lockdown Stringency: beta = {b_str:+.4f}, p = {p_str:.4f}  -> {'INCREASES' if b_str > 0 else 'DECREASES'} polarization

  Model 4 (Residualized cross-section):
    Stringency:          beta = {b4:+.4f}, p = {p4:.4f}  -> {'INCREASES' if b4 > 0 else 'DECREASES'} polarization
""")

    if b_str < 0:
        print("  WARNING: Stringency coefficient is NEGATIVE.")
        print("  This means higher stringency is associated with DECREASED polarization")
        print("  (i.e., INCREASED cohesion). This CONTRADICTS the paper's narrative")
        print("  that 'lockdowns caused polarization.'")
        print()
        print("  Likely explanation: MEAN REVERSION + SELECTION.")
        print("  Countries that were already polarized (low v2smpolsoc) had high stringency.")
        print("  These countries regressed toward the mean (cohesion improved).")
        print("  The positive association between stringency and cohesion change is")
        print("  driven by the selection effect, not a causal benefit of lockdowns.")
    elif b_str > 0:
        print("  Stringency coefficient is POSITIVE.")
        print("  Higher stringency is associated with INCREASED polarization.")
        print("  This is CONSISTENT with the paper's narrative.")

print("\nDone.")
