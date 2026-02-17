#!/usr/bin/env python3
"""
EIPSA — 03: Robustness Checks (Supplementary Information)
===========================================================
Reproduces all SI analyses from:
  "The Stress Test That Changed Nothing"

SI-1: Economic Buffer Hypothesis (OxCGRT Economic Support Index)
SI-2: GDP control for the Selection Effect
SI-3: Sensitivity Analysis
  - Alternative outcomes (v2cacamps, v2x_libdem, v2x_jucon)
  - Lagged stringency (t-1, t-2)
  - Jackknife leave-one-out for the selection effect
SI-4: Forest plot (SI Figure S1)

Outputs:
  - output/figures/si_figure_s1.png
  - output/tables/si_robustness.txt
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
import matplotlib.patches as mpatches

# ── Paths (relative to this script) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(SCRIPT_DIR, os.pardir)
DATA_PATH  = os.path.join(ROOT_DIR, "data", "EIPSA_OECD_panel_2019_2024.csv")
FIG_DIR    = os.path.join(ROOT_DIR, "output", "figures")
TBL_DIR    = os.path.join(ROOT_DIR, "output", "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)


def sig(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


# =================================================================
# 1. DATA LOADING & PREPARATION
# =================================================================
print("=" * 78)
print(" EIPSA — 03: Robustness Checks")
print("=" * 78)

df = pd.read_csv(DATA_PATH)

# ── Panel (2015–2024) ──
panel = df[df["year"] >= 2015].copy().sort_values(["iso3", "year"])

# Outcome variables (positive = more polarized / less democratic)
panel["polarization"] = -panel["v2smpolsoc"]
panel["pol_camps"]    =  panel["v2cacamps"]
panel["libdem"]       =  panel["v2x_libdem"]
panel["jucon"]        =  panel["v2x_jucon"]

for var in ["polarization", "pol_camps", "libdem", "jucon"]:
    panel[f"d_{var}"] = panel.groupby("iso3")[var].diff()

# Lagged stringency
panel["stringency_norm_L1"] = panel.groupby("iso3")["stringency_norm"].shift(1)
panel["stringency_norm_L2"] = panel.groupby("iso3")["stringency_norm"].shift(2)

# Interaction
panel["string_x_econ"] = panel["stringency_norm"] * panel["econ_support_norm"]

panel = panel.set_index(["iso3", "year"])

# ── Cross-section ──
cs = (
    df[["iso3", "country_name", "stringency_avg_2020_2021",
        "econ_support_avg_2020_2021", "gdp_per_capita_2019"]]
    .drop_duplicates("iso3").copy()
)
coh_2019 = df[df["year"] == 2019][["iso3", "v2smpolsoc"]].rename(
    columns={"v2smpolsoc": "cohesion_2019"})
coh_2024 = df[df["year"] == 2024][["iso3", "v2smpolsoc"]].rename(
    columns={"v2smpolsoc": "cohesion_2024"})
cs = cs.merge(coh_2019, on="iso3", how="left")
cs = cs.merge(coh_2024, on="iso3", how="left")
cs = cs.dropna(subset=["cohesion_2019", "stringency_avg_2020_2021"])
cs = cs.reset_index(drop=True)
cs["d_pol"] = -(cs["cohesion_2024"] - cs["cohesion_2019"])
cs["log_gdppc"] = np.log(cs["gdp_per_capita_2019"])

print(f"  Panel: {len(panel)} obs | Cross-section: {len(cs)} countries")

# Collector for forest plot
forest_results = []
# Also collect all text output for the SI table file
si_lines = []

def siprint(line=""):
    print(line)
    si_lines.append(line)


# =================================================================
# SI-1: ECONOMIC BUFFER HYPOTHESIS
# =================================================================
siprint(f"\n{'=' * 78}")
siprint(" SI-1: Economic Buffer Hypothesis")
siprint("=" * 78)

# Model A: Additive
cols_a = ["stringency_norm", "econ_support_norm"]
valid_a = panel[["d_polarization"] + cols_a].dropna()
m_a = PanelOLS(
    valid_a["d_polarization"], valid_a[cols_a],
    entity_effects=True, time_effects=True,
).fit(cov_type="clustered", cluster_entity=True)

siprint("  Model A (additive):")
for var in cols_a:
    b, se_, p_ = m_a.params[var], m_a.std_errors[var], m_a.pvalues[var]
    ci = m_a.conf_int().loc[var]
    siprint(f"    {var:>25s}: b = {b:+.4f}{sig(p_)}, SE = {se_:.4f}, "
            f"95% CI [{ci['lower']:+.4f}, {ci['upper']:+.4f}], p = {p_:.4f}")

forest_results.append({
    "label": "With Economic Controls",
    "b": m_a.params["stringency_norm"],
    "ci_lo": m_a.conf_int().loc["stringency_norm", "lower"],
    "ci_hi": m_a.conf_int().loc["stringency_norm", "upper"],
    "p": m_a.pvalues["stringency_norm"],
    "group": "robustness",
})

# Model B: With interaction
cols_b = ["stringency_norm", "econ_support_norm", "string_x_econ"]
valid_b = panel[["d_polarization"] + cols_b].dropna()
m_b = PanelOLS(
    valid_b["d_polarization"], valid_b[cols_b],
    entity_effects=True, time_effects=True,
).fit(cov_type="clustered", cluster_entity=True)

siprint("\n  Model B (with interaction):")
for var in cols_b:
    b, se_, p_ = m_b.params[var], m_b.std_errors[var], m_b.pvalues[var]
    ci = m_b.conf_int().loc[var]
    siprint(f"    {var:>25s}: b = {b:+.4f}{sig(p_)}, SE = {se_:.4f}, "
            f"95% CI [{ci['lower']:+.4f}, {ci['upper']:+.4f}], p = {p_:.4f}")


# =================================================================
# SI-2: GDP CONTROL FOR SELECTION EFFECT
# =================================================================
siprint(f"\n{'=' * 78}")
siprint(" SI-2: GDP Control for Selection Effect")
siprint("=" * 78)

cs_gdp = cs.dropna(subset=["log_gdppc"]).reset_index(drop=True)

X1 = sm.add_constant(cs_gdp["cohesion_2019"])
m_sel1 = sm.OLS(cs_gdp["stringency_avg_2020_2021"], X1).fit(cov_type="HC1")
siprint(f"  Bivariate: b = {m_sel1.params['cohesion_2019']:+.3f}, "
        f"p = {m_sel1.pvalues['cohesion_2019']:.4f}, "
        f"R2 = {m_sel1.rsquared:.3f}")

X2 = sm.add_constant(cs_gdp[["cohesion_2019", "log_gdppc"]])
m_sel2 = sm.OLS(cs_gdp["stringency_avg_2020_2021"], X2).fit(cov_type="HC1")
siprint(f"  With GDP:  Cohesion b = {m_sel2.params['cohesion_2019']:+.3f}, "
        f"p = {m_sel2.pvalues['cohesion_2019']:.4f}")
siprint(f"             GDP      b = {m_sel2.params['log_gdppc']:+.3f}, "
        f"p = {m_sel2.pvalues['log_gdppc']:.4f}, "
        f"R2 = {m_sel2.rsquared:.3f}")

# Partial correlation
resid_coh = sm.OLS(cs_gdp["cohesion_2019"],
                   sm.add_constant(cs_gdp["log_gdppc"])).fit().resid
resid_str = sm.OLS(cs_gdp["stringency_avg_2020_2021"],
                   sm.add_constant(cs_gdp["log_gdppc"])).fit().resid
r_partial, p_partial = sp_stats.pearsonr(resid_coh, resid_str)
siprint(f"  Partial r (cohesion-stringency | GDP): "
        f"{r_partial:.3f}, p = {p_partial:.4f}")


# =================================================================
# SI-3a: ALTERNATIVE OUTCOMES
# =================================================================
siprint(f"\n{'=' * 78}")
siprint(" SI-3a: Alternative Outcomes")
siprint("=" * 78)

outcomes = [
    ("d_polarization", "Social Polarization (v2smpolsoc)", "main"),
    ("d_pol_camps",    "Political Polarization (v2cacamps)", "robustness"),
    ("d_libdem",       "Liberal Democracy (v2x_libdem)", "robustness"),
    ("d_jucon",        "Judicial Constraints (v2x_jucon)", "robustness"),
]

for outcome_var, label, group in outcomes:
    cols = ["covid_intensity", "stringency_norm"]
    valid = panel[[outcome_var] + cols].dropna()
    if len(valid) < 50:
        continue
    try:
        m = PanelOLS(valid[outcome_var], valid[cols],
                     entity_effects=True, time_effects=True,
                     ).fit(cov_type="clustered", cluster_entity=True)
        ci = m.conf_int().loc["stringency_norm"]
        siprint(f"  {label:<42s}: b = {m.params['stringency_norm']:+.4f}"
                f"{sig(m.pvalues['stringency_norm'])}, "
                f"p = {m.pvalues['stringency_norm']:.4f}")
        forest_results.append({
            "label": f"Alt: {label.split('(')[0].strip()}" if group == "robustness"
                     else "Main Model (Social Polarization)",
            "b": m.params["stringency_norm"],
            "ci_lo": ci["lower"], "ci_hi": ci["upper"],
            "p": m.pvalues["stringency_norm"],
            "group": group,
        })
    except Exception as e:
        siprint(f"  {label}: FAILED \u2014 {e}")


# =================================================================
# SI-3b: LAGGED STRINGENCY
# =================================================================
siprint(f"\n{'=' * 78}")
siprint(" SI-3b: Lagged Stringency")
siprint("=" * 78)

for lag_var, label in [("stringency_norm",    "Contemporaneous (t)"),
                       ("stringency_norm_L1", "Lag t\u22121"),
                       ("stringency_norm_L2", "Lag t\u22122")]:
    valid = panel[["d_polarization", lag_var]].dropna()
    if len(valid) < 50:
        continue
    try:
        m = PanelOLS(valid["d_polarization"], valid[[lag_var]],
                     entity_effects=True, time_effects=True,
                     ).fit(cov_type="clustered", cluster_entity=True)
        ci = m.conf_int().loc[lag_var]
        siprint(f"  {label:<22s}: b = {m.params[lag_var]:+.4f}"
                f"{sig(m.pvalues[lag_var])}, p = {m.pvalues[lag_var]:.4f}")
        if "Contemporaneous" not in label:
            forest_results.append({
                "label": f"Lagged Stringency ({label.split(' ')[1]})",
                "b": m.params[lag_var],
                "ci_lo": ci["lower"], "ci_hi": ci["upper"],
                "p": m.pvalues[lag_var],
                "group": "robustness",
            })
    except Exception as e:
        siprint(f"  {label}: FAILED \u2014 {e}")


# =================================================================
# SI-3c: JACKKNIFE (Leave-One-Out)
# =================================================================
siprint(f"\n{'=' * 78}")
siprint(" SI-3c: Jackknife \u2014 Selection Effect")
siprint("=" * 78)

r_full, p_full = sp_stats.pearsonr(
    cs["cohesion_2019"], cs["stringency_avg_2020_2021"])
siprint(f"  Full sample: r = {r_full:.4f}, p = {p_full:.4f}, N = {len(cs)}")

jk = []
for idx in cs.index:
    tmp = cs.drop(idx)
    r_loo, p_loo = sp_stats.pearsonr(
        tmp["cohesion_2019"], tmp["stringency_avg_2020_2021"])
    jk.append({"dropped": cs.loc[idx, "iso3"],
               "country": cs.loc[idx, "country_name"],
               "r": r_loo, "p": p_loo})
jk = pd.DataFrame(jk)

siprint(f"  Range: [{jk['r'].min():.4f}, {jk['r'].max():.4f}]")
siprint(f"  Mean:  {jk['r'].mean():.4f}, SD: {jk['r'].std():.4f}")
siprint(f"  All p < 0.01: {(jk['p'] < 0.01).all()}")

siprint(f"\n  {'Dropped':>6s}  {'Country':<30s}  {'r':>8s}  {'p':>8s}")
siprint(f"  {'-'*6}  {'-'*30}  {'-'*8}  {'-'*8}")
for _, row in jk.sort_values("r").iterrows():
    siprint(f"  {row['dropped']:>6s}  {row['country']:<30s}  "
            f"{row['r']:+.4f}  {row['p']:.4f}")


# =================================================================
# SI-4: FOREST PLOT (SI Figure S1)
# =================================================================
siprint(f"\n{'=' * 78}")
siprint(" SI Figure S1: Forest Plot")
siprint("=" * 78)

fr = pd.DataFrame(forest_results)
order = [
    "Main Model (Social Polarization)",
    "Alt: Political Polarization",
    "Alt: Liberal Democracy",
    "Alt: Judicial Constraints",
    "With Economic Controls",
    "Lagged Stringency (t\u22121)",
    "Lagged Stringency (t\u22122)",
]
fr["label"] = pd.Categorical(
    fr["label"],
    categories=[o for o in order if o in fr["label"].values],
    ordered=True)
fr = fr.sort_values("label", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = {"main": "#2166ac", "robustness": "#666666"}

for i, row in fr.iterrows():
    color = colors[row["group"]]
    ax.errorbar(row["b"], i,
                xerr=[[row["b"] - row["ci_lo"]], [row["ci_hi"] - row["b"]]],
                fmt="o", color=color, markersize=8, capsize=5, capthick=1.5,
                elinewidth=1.5, markeredgecolor="white", markeredgewidth=0.8,
                zorder=4)

ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7, zorder=2)

x_max = max(abs(fr["ci_lo"].min()), abs(fr["ci_hi"].max())) * 1.3
ax.axvspan(0, x_max, alpha=0.04, color="#d6604d", zorder=0)
ax.axvspan(-x_max, 0, alpha=0.04, color="#4393c3", zorder=0)

ax.text(x_max * 0.7, -0.8, "Lockdowns increase\npolarization",
        fontsize=8, fontstyle="italic", color="#b2182b", alpha=0.5,
        ha="center", va="top")
ax.text(-x_max * 0.7, -0.8, "Lockdowns decrease\npolarization",
        fontsize=8, fontstyle="italic", color="#2166ac", alpha=0.5,
        ha="center", va="top")

ax.set_yticks(range(len(fr)))
ax.set_yticklabels(fr["label"], fontsize=10)
ax.set_xlabel("Coefficient Estimate of Lockdown Stringency\n"
              "(with 95% Confidence Interval)", fontsize=11, labelpad=10)
ax.set_title("SI Figure S1.  Stringency Coefficients Across All "
             "Specifications\nTWFE First-Difference Models, "
             "38 OECD Countries (2015\u20132024)",
             fontsize=12, fontweight="bold", pad=14)
ax.set_xlim(-x_max, x_max)
ax.grid(axis="x", alpha=0.25, color="#cccccc")
ax.set_axisbelow(True)

main_patch  = mpatches.Patch(color="#2166ac", label="Main specification")
robust_patch = mpatches.Patch(color="#666666", label="Robustness check")
ax.legend(handles=[main_patch, robust_patch], loc="lower right",
          fontsize=9, framealpha=0.9)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig_path = os.path.join(FIG_DIR, "si_figure_s1.png")
fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
siprint(f"  Saved -> {fig_path}")
plt.close(fig)


# =================================================================
# SUMMARY TABLE
# =================================================================
siprint(f"\n{'=' * 78}")
siprint(" Summary: All Stringency Coefficients")
siprint("=" * 78)

siprint(f"\n  {'Specification':<40s}  {'Coeff':>8s}  "
        f"{'95% CI':>20s}  {'p':>8s}  {'Sig':>5s}")
siprint(f"  {'-'*40}  {'-'*8}  {'-'*20}  {'-'*8}  {'-'*5}")
for _, row in fr.sort_values("label").iterrows():
    ci_str = f"[{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]"
    siprint(f"  {row['label']:<40s}  {row['b']:+.4f}  "
            f"{ci_str:>20s}  {row['p']:.4f}  {sig(row['p']):>5s}")

any_pos_sig = ((fr["b"] > 0) & (fr["p"] < 0.05)).any()
siprint(f"\n  Significant positive stringency effect in any spec: {any_pos_sig}")

# Save SI output
si_path = os.path.join(TBL_DIR, "si_robustness.txt")
with open(si_path, "w") as f:
    f.write("\n".join(si_lines))
siprint(f"\n  Saved -> {si_path}")
siprint("\n  Done.")
