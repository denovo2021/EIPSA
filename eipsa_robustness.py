#!/usr/bin/env python3
"""
EIPSA – Robustness: Reverse Causality & Interaction Analysis
==============================================================
Checks whether the "Stringency → Social Polarization" finding
is driven by reverse causality (already-polarized countries
chose stricter lockdowns) and whether pre-existing social trust
moderates the effect.

Requirements:
    pip install pandas openpyxl linearmodels matplotlib scipy
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from linearmodels.panel import PanelOLS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    if p < 0.10:  return "†"
    return ""


# =====================================================================
# 1.  LOAD & BUILD CROSS-SECTION
# =====================================================================
print("=" * 78)
print(" 1.  DATA LOADING")
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
        stringency_max=("stringency_index", "max"),
    )
    .rename(columns={"iso_code": "iso3"})
)

# Mean stringency 2020-2021
str_2021 = (
    owid_cy[(owid_cy["iso3"].isin(OECD)) & (owid_cy["year"].isin([2020, 2021]))]
    .groupby("iso3")
    .agg(
        stringency_avg=("stringency_mean", "mean"),
        stringency_peak=("stringency_max", "max"),
    )
    .reset_index()
)

# ── V-Dem ──
vdem_cols = [
    "country_text_id", "country_name", "year",
    "v2smpolsoc",   # social polarization (outcome)
    "v2cacamps",    # political polarization
    "v2x_cspart",   # civil society participation (trust proxy)
    "v2dlengage",   # engaged society (trust proxy)
    "v2xdl_delib",  # deliberative component (trust proxy)
    "v2x_libdem",   # liberal democracy (context)
]
vdem = pd.read_csv(VDEM_PATH, usecols=vdem_cols, low_memory=False)
vdem = vdem.rename(columns={"country_text_id": "iso3"})

# Build cross-section: one row per OECD country
cs = pd.DataFrame({"iso3": OECD})

# Merge country names
names = (
    vdem[vdem["iso3"].isin(OECD)]
    .drop_duplicates("iso3")[["iso3", "country_name"]]
)
cs = cs.merge(names, on="iso3", how="left")

# Merge stringency
cs = cs.merge(str_2021, on="iso3", how="left")

# Merge V-Dem at key years
for yr in [2019, 2022, 2024]:
    yr_data = vdem[vdem["year"] == yr][
        ["iso3", "v2smpolsoc", "v2cacamps",
         "v2x_cspart", "v2dlengage", "v2xdl_delib", "v2x_libdem"]
    ]
    yr_data = yr_data.rename(
        columns={c: f"{c}_{yr}" for c in yr_data.columns if c != "iso3"}
    )
    cs = cs.merge(yr_data, on="iso3", how="left")

# Compute changes
cs["d_polsoc_19_24"] = cs["v2smpolsoc_2024"] - cs["v2smpolsoc_2019"]
cs["d_polsoc_19_22"] = cs["v2smpolsoc_2022"] - cs["v2smpolsoc_2019"]
cs["d_camps_19_24"]  = cs["v2cacamps_2024"]  - cs["v2cacamps_2019"]

# COVID deaths
covid_2021 = owid_cy[
    (owid_cy["iso3"].isin(OECD)) & (owid_cy["year"] == 2021)
][["iso3", "covid_deaths_pm_eoy"]].rename(
    columns={"covid_deaths_pm_eoy": "covid_pm_2021"}
)
cs = cs.merge(covid_2021, on="iso3", how="left")
cs["ln_covid"] = np.log1p(cs["covid_pm_2021"].fillna(0))

print(f"  Cross-section: {len(cs)} OECD countries")


# =====================================================================
# 2.  REVERSE CAUSALITY CHECK
# =====================================================================
print(f"\n\n{'=' * 78}")
print(" 2.  REVERSE CAUSALITY CHECK")
print("    Q: Did already-polarized countries choose stricter lockdowns?")
print("=" * 78)

checks = [
    ("v2smpolsoc_2019", "Social Polarization (2019)",
     "stringency_avg",  "Mean Stringency (2020\u201321)"),
    ("v2cacamps_2019",  "Political Polarization (2019)",
     "stringency_avg",  "Mean Stringency (2020\u201321)"),
    ("v2x_cspart_2019", "Civil Society Participation (2019)",
     "stringency_avg",  "Mean Stringency (2020\u201321)"),
    ("v2dlengage_2019", "Engaged Society (2019)",
     "stringency_avg",  "Mean Stringency (2020\u201321)"),
    ("v2xdl_delib_2019","Deliberative Component (2019)",
     "stringency_avg",  "Mean Stringency (2020\u201321)"),
    ("v2x_libdem_2019", "Liberal Democracy (2019)",
     "stringency_avg",  "Mean Stringency (2020\u201321)"),
]

print(f"\n  {'Pre-COVID variable (2019)':<40s} {'vs. Stringency':>18s} "
      f"{'r':>7s} {'R²':>7s} {'p':>8s}")
print("  " + "─" * 85)

reverse_results = {}
for xvar, xlabel, yvar, ylabel in checks:
    valid = cs.dropna(subset=[xvar, yvar])
    slope, intercept, r, p, se = sp_stats.linregress(valid[xvar], valid[yvar])
    reverse_results[xvar] = (r, r**2, p, slope)
    print(f"  {xlabel:<40s} {ylabel:>18s} "
          f"{r:>+7.3f} {r**2:>7.3f} {p:>8.4f}{sig(p)}")

# Also check: did pre-2019 social polarization predict COVID deaths?
valid = cs.dropna(subset=["v2smpolsoc_2019", "ln_covid"])
slope, intercept, r, p, se = sp_stats.linregress(
    valid["v2smpolsoc_2019"], valid["ln_covid"]
)
print(f"\n  {'Social Polarization (2019)':<40s} {'ln(COVID deaths/M)':>18s} "
      f"{r:>+7.3f} {r**2:>7.3f} {p:>8.4f}{sig(p)}")


# ── Reverse causality scatter (3 panels) ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

scatter_rev = [
    ("v2smpolsoc_2019", "Social Polarization (2019)",
     "stringency_avg",  "Mean Stringency 2020\u201321"),
    ("v2cacamps_2019",  "Political Polarization (2019)",
     "stringency_avg",  "Mean Stringency 2020\u201321"),
    ("v2x_libdem_2019", "Liberal Democracy (2019)",
     "stringency_avg",  "Mean Stringency 2020\u201321"),
]

for ax, (xvar, xlabel, yvar, ylabel) in zip(axes, scatter_rev):
    valid = cs.dropna(subset=[xvar, yvar])
    ax.scatter(valid[xvar], valid[yvar], s=50, alpha=0.7,
               edgecolors="k", linewidths=0.5, zorder=3)
    for _, row in valid.iterrows():
        ax.annotate(row["iso3"], (row[xvar], row[yvar]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")

    slope, intercept, r, p, se = sp_stats.linregress(valid[xvar], valid[yvar])
    xline = np.linspace(valid[xvar].min(), valid[xvar].max(), 100)
    ax.plot(xline, intercept + slope * xline, "r--", linewidth=1.5, alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"r = {r:+.3f},  R² = {r**2:.3f},  p = {p:.4f}",
                 fontsize=10)

fig.suptitle(
    "Reverse causality check: did pre-COVID conditions predict lockdown stringency?\n"
    "(each dot = one OECD country)",
    fontsize=12, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig("eipsa_reverse_causality.png", dpi=200, bbox_inches="tight")
print(f"\n  Reverse causality plot saved → eipsa_reverse_causality.png")
plt.close()


# =====================================================================
# 3.  INTERACTION MODEL — Does pre-existing trust moderate the effect?
# =====================================================================
print(f"\n\n{'=' * 78}")
print(" 3.  INTERACTION MODEL — Trust × Stringency")
print("    Q: Were high-trust societies immune to lockdown-driven polarization?")
print("=" * 78)

# Cross-sectional interaction: ΔPolSoc = β₁·Stringency + β₂·Trust₂₀₁₉
#                                       + β₃·(Stringency × Trust) + ε
# β₃ < 0 → high trust buffers the polarizing effect of lockdowns

trust_vars = [
    ("v2x_cspart_2019",  "Civil Society Participation"),
    ("v2dlengage_2019",  "Engaged Society"),
    ("v2xdl_delib_2019", "Deliberative Component"),
]

for dep, dep_label in [("d_polsoc_19_24", "Δ Social Polarization 2019→2024"),
                         ("d_polsoc_19_22", "Δ Social Polarization 2019→2022")]:
    print(f"\n  Dependent variable: {dep_label}")
    print(f"  {'Trust variable':<35s} {'β(Stringency)':>15s} "
          f"{'β(Interaction)':>15s} {'p(interaction)':>15s} {'R²':>7s}")
    print("  " + "─" * 90)

    for tvar, tlabel in trust_vars:
        valid = cs.dropna(subset=["stringency_avg", tvar, dep])
        if len(valid) < 10:
            continue

        # Standardise for interpretability
        S = (valid["stringency_avg"] - valid["stringency_avg"].mean()) / \
            valid["stringency_avg"].std()
        T = (valid[tvar] - valid[tvar].mean()) / valid[tvar].std()
        Y = valid[dep].values
        interaction = S * T

        import statsmodels.api as sm
        X = sm.add_constant(pd.DataFrame({
            "Stringency": S.values,
            "Trust": T.values,
            "S_x_T": interaction.values,
        }))
        model = sm.OLS(Y, X).fit(cov_type="HC1")

        b_s = model.params["Stringency"]
        b_int = model.params["S_x_T"]
        p_int = model.pvalues["S_x_T"]
        r2 = model.rsquared

        print(f"  {tlabel:<35s} {b_s:>+15.4f} "
              f"{b_int:>+15.4f} {p_int:>15.4f}{sig(p_int)} {r2:>7.3f}")

    # Print the full model for the best trust proxy
    print(f"\n  Full model: {dep_label} ~ Stringency + CivSoc + Stringency×CivSoc")
    valid = cs.dropna(subset=["stringency_avg", "v2x_cspart_2019", dep])
    S = (valid["stringency_avg"] - valid["stringency_avg"].mean()) / \
        valid["stringency_avg"].std()
    T = (valid["v2x_cspart_2019"] - valid["v2x_cspart_2019"].mean()) / \
        valid["v2x_cspart_2019"].std()
    Y = valid[dep].values
    X = sm.add_constant(pd.DataFrame({
        "Stringency": S.values,
        "CivSoc_2019": T.values,
        "Stringency_x_CivSoc": (S * T).values,
    }))
    model = sm.OLS(Y, X).fit(cov_type="HC1")
    print(model.summary2().tables[1].to_string())


# =====================================================================
# 4.  FINAL VISUALIZATION — labelled scatter + residual plot
# =====================================================================
print(f"\n\n{'=' * 78}")
print(" 4.  FINAL VISUALIZATIONS")
print("=" * 78)

fig = plt.figure(figsize=(20, 14))

# ── Panel A: Stringency vs ΔSocialPolarization (labelled) ──
ax1 = fig.add_subplot(2, 2, 1)
valid = cs.dropna(subset=["stringency_avg", "d_polsoc_19_24"])
slope, intercept, r, p, se = sp_stats.linregress(
    valid["stringency_avg"], valid["d_polsoc_19_24"]
)
ax1.scatter(valid["stringency_avg"], valid["d_polsoc_19_24"],
            s=60, alpha=0.7, c="#1f77b4", edgecolors="k", linewidths=0.5,
            zorder=3)
for _, row in valid.iterrows():
    ax1.annotate(
        row["iso3"],
        (row["stringency_avg"], row["d_polsoc_19_24"]),
        fontsize=7.5, fontweight="bold",
        ha="left", va="bottom",
        xytext=(4, 4), textcoords="offset points",
    )
xline = np.linspace(valid["stringency_avg"].min(),
                     valid["stringency_avg"].max(), 100)
ax1.plot(xline, intercept + slope * xline, "r-", linewidth=2, alpha=0.8)
ax1.fill_between(
    xline,
    intercept + (slope - 1.96 * se) * xline,
    intercept + (slope + 1.96 * se) * xline,
    alpha=0.1, color="red",
)
ax1.axhline(0, color="grey", linewidth=0.8, linestyle=":")
ax1.set_xlabel("Mean Stringency Index (2020\u201321)", fontsize=11)
ax1.set_ylabel("Δ Social Polarization (2019 → 2024)", fontsize=11)
ax1.set_title(
    f"A. Lockdown stringency vs. social polarization change\n"
    f"slope = {slope:+.4f}, R² = {r**2:.3f}, p = {p:.4f}{sig(p)}",
    fontsize=11, fontweight="bold",
)

# ── Panel B: Reverse causality (2019 level → stringency) ──
ax2 = fig.add_subplot(2, 2, 2)
valid2 = cs.dropna(subset=["v2smpolsoc_2019", "stringency_avg"])
slope2, int2, r2, p2, se2 = sp_stats.linregress(
    valid2["v2smpolsoc_2019"], valid2["stringency_avg"]
)
ax2.scatter(valid2["v2smpolsoc_2019"], valid2["stringency_avg"],
            s=60, alpha=0.7, c="#2ca02c", edgecolors="k", linewidths=0.5,
            zorder=3)
for _, row in valid2.iterrows():
    ax2.annotate(row["iso3"],
                 (row["v2smpolsoc_2019"], row["stringency_avg"]),
                 fontsize=7.5, fontweight="bold",
                 ha="left", va="bottom",
                 xytext=(4, 4), textcoords="offset points")
xline2 = np.linspace(valid2["v2smpolsoc_2019"].min(),
                      valid2["v2smpolsoc_2019"].max(), 100)
ax2.plot(xline2, int2 + slope2 * xline2, "r--", linewidth=1.5, alpha=0.8)
ax2.set_xlabel("Social Polarization level (2019, pre-COVID)", fontsize=11)
ax2.set_ylabel("Mean Stringency Index (2020\u201321)", fontsize=11)
ax2.set_title(
    f"B. Reverse causality check: pre-COVID polarization → stringency\n"
    f"r = {r2:+.3f}, R² = {r2**2:.3f}, p = {p2:.4f}  "
    f"({'CONCERN' if p2 < 0.1 else 'No evidence'})",
    fontsize=11, fontweight="bold",
)

# ── Panel C: Residualized plot (partial out 2019 baseline) ──
ax3 = fig.add_subplot(2, 2, 3)
valid3 = cs.dropna(subset=["stringency_avg", "d_polsoc_19_24",
                            "v2smpolsoc_2019"])
# Residualise: remove effect of 2019 baseline from both X and Y
import statsmodels.api as sm

X_base = sm.add_constant(valid3["v2smpolsoc_2019"])
resid_y = sm.OLS(valid3["d_polsoc_19_24"], X_base).fit().resid
resid_x = sm.OLS(valid3["stringency_avg"], X_base).fit().resid

slope3, int3, r3, p3, se3 = sp_stats.linregress(resid_x, resid_y)
ax3.scatter(resid_x, resid_y, s=60, alpha=0.7, c="#d62728",
            edgecolors="k", linewidths=0.5, zorder=3)
for idx, row in valid3.iterrows():
    i = valid3.index.get_loc(idx)
    ax3.annotate(row["iso3"], (resid_x.iloc[i], resid_y.iloc[i]),
                 fontsize=7.5, fontweight="bold",
                 ha="left", va="bottom",
                 xytext=(4, 4), textcoords="offset points")
xline3 = np.linspace(resid_x.min(), resid_x.max(), 100)
ax3.plot(xline3, int3 + slope3 * xline3, "r-", linewidth=2, alpha=0.8)
ax3.axhline(0, color="grey", linewidth=0.8, linestyle=":")
ax3.axvline(0, color="grey", linewidth=0.8, linestyle=":")
ax3.set_xlabel("Stringency (residualized: ⊥ 2019 polarization)",
               fontsize=11)
ax3.set_ylabel("Δ Social Polarization (residualized: ⊥ 2019 baseline)",
               fontsize=11)
ax3.set_title(
    f"C. Partial regression: Stringency → ΔPolSoc | baseline\n"
    f"slope = {slope3:+.4f}, R² = {r3**2:.3f}, p = {p3:.4f}{sig(p3)}",
    fontsize=11, fontweight="bold",
)

# ── Panel D: Interaction — split by high/low trust ──
ax4 = fig.add_subplot(2, 2, 4)
valid4 = cs.dropna(subset=["stringency_avg", "d_polsoc_19_24",
                            "v2x_cspart_2019"])
median_trust = valid4["v2x_cspart_2019"].median()
high_trust = valid4[valid4["v2x_cspart_2019"] >= median_trust]
low_trust  = valid4[valid4["v2x_cspart_2019"] < median_trust]

for group, label, color, marker in [
    (high_trust, "High civil society (above median)", "#1f77b4", "o"),
    (low_trust,  "Low civil society (below median)",  "#d62728", "s"),
]:
    ax4.scatter(group["stringency_avg"], group["d_polsoc_19_24"],
                s=60, alpha=0.7, c=color, edgecolors="k", linewidths=0.5,
                marker=marker, label=label, zorder=3)
    for _, row in group.iterrows():
        ax4.annotate(row["iso3"],
                     (row["stringency_avg"], row["d_polsoc_19_24"]),
                     fontsize=7, ha="left", va="bottom",
                     xytext=(3, 3), textcoords="offset points")
    if len(group) >= 5:
        sl, it, r_, p_, _ = sp_stats.linregress(
            group["stringency_avg"], group["d_polsoc_19_24"]
        )
        xl_ = np.linspace(group["stringency_avg"].min(),
                          group["stringency_avg"].max(), 50)
        ax4.plot(xl_, it + sl * xl_, "--", color=color, linewidth=1.5,
                 alpha=0.8)

ax4.axhline(0, color="grey", linewidth=0.8, linestyle=":")
ax4.set_xlabel("Mean Stringency Index (2020\u201321)", fontsize=11)
ax4.set_ylabel("Δ Social Polarization (2019 → 2024)", fontsize=11)
ax4.set_title(
    "D. Interaction: does civil society participation moderate the effect?",
    fontsize=11, fontweight="bold",
)
ax4.legend(fontsize=9, loc="upper left")

fig.suptitle(
    "EIPSA: Lockdown stringency and social polarization in OECD countries\n"
    "Robustness checks for reverse causality and trust moderation",
    fontsize=14, fontweight="bold", y=0.98,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("eipsa_final_robustness.png", dpi=200, bbox_inches="tight")
print(f"  Final plot saved → eipsa_final_robustness.png")
plt.close()


# =====================================================================
# 5.  SUMMARY
# =====================================================================
print(f"\n\n{'=' * 78}")
print(" 5.  SUMMARY OF ROBUSTNESS CHECKS")
print("=" * 78)

r_rev = reverse_results["v2smpolsoc_2019"]
print(f"""
  REVERSE CAUSALITY:
    Social Polarization (2019) vs. Stringency (2020-21):
      r = {r_rev[0]:+.3f},  R² = {r_rev[1]:.3f},  p = {r_rev[2]:.4f}
""")

if r_rev[2] > 0.10:
    print("    → NO evidence of reverse causality.")
    print("      Pre-COVID social polarization does NOT predict lockdown")
    print("      stringency. Countries did not lock down harder *because*")
    print("      they were already polarized.")
elif r_rev[2] > 0.05:
    print("    → WEAK evidence of reverse causality (marginal).")
    print("      Some concern, but the residualized analysis (Panel C)")
    print("      can address this.")
else:
    print("    → EVIDENCE of reverse causality.")
    print("      The residualized plot (Panel C) is essential to check")
    print("      whether the effect survives after partialling out baseline.")

print(f"""
  RESIDUALIZED RELATIONSHIP (Panel C):
    Stringency → ΔSocialPolarization | baseline 2019:
      slope = {slope3:+.4f},  R² = {r3**2:.3f},  p = {p3:.4f}{sig(p3)}
""")

if p3 < 0.05:
    print("    → The effect SURVIVES after controlling for baseline.")
    print("      Lockdown stringency predicts change in social polarization")
    print("      even after removing the influence of pre-existing polarization.")
elif p3 < 0.10:
    print("    → The effect is MARGINAL after controlling for baseline.")
else:
    print("    → The effect WEAKENS after controlling for baseline.")

print(f"""
  INTERACTION (Trust moderation):
    If the Stringency × CivilSociety interaction term is negative,
    it means high-trust countries are buffered from the polarizing
    effect of lockdowns.
""")

print("Done.")
