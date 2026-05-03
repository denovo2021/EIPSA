"""
final_outputs.py
================
Loads data and InferenceData from disk, exports 95% HDI posterior summaries,
and regenerates Figures 1 and 2 for the EIPSA manuscript.

Run:    uv run python scripts/final_outputs.py
"""

from pathlib import Path
import sys

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent.parent
PANEL_CSV    = ROOT / "data" / "oecd_panel.csv"
IDATA_MAIN   = ROOT / "output" / "tables" / "bayesian_idata.nc"
IDATA_LAG2   = ROOT / "output" / "tables" / "bayesian_idata_lag2.nc"   # optional
IDATA_INTER  = ROOT / "output" / "tables" / "bayesian_idata_inter.nc"  # optional
TABLES_DIR   = ROOT / "output" / "tables"
FIGURES_DIR  = ROOT / "output" / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. Load panel and InferenceData objects
# -----------------------------------------------------------------------------
print("Loading data ...")
df_panel = pd.read_csv(PANEL_CSV)

if not IDATA_MAIN.exists():
    sys.exit(f"ERROR: primary InferenceData not found at {IDATA_MAIN}")

idata_main = az.from_netcdf(IDATA_MAIN)
idata_lag2  = az.from_netcdf(IDATA_LAG2)  if IDATA_LAG2.exists()  else None
idata_inter = az.from_netcdf(IDATA_INTER) if IDATA_INTER.exists() else None

print(f"  primary idata : {IDATA_MAIN.name}")
print(f"  lag-2 idata   : "
      f"{IDATA_LAG2.name if idata_lag2 is not None else '(missing - skipping)'}")
print(f"  interact idata: "
      f"{IDATA_INTER.name if idata_inter is not None else '(missing - skipping)'}")


# -----------------------------------------------------------------------------
# 2. Posterior summary tables at 95% HDI
# -----------------------------------------------------------------------------
def export_summary(idata, label: str) -> None:
    if idata is None:
        print(f"  skipping {label} (InferenceData not on disk)")
        return
    summary = az.summary(idata, hdi_prob=0.95, stat_focus="mean", round_to="none")
    out = TABLES_DIR / f"posterior_summary_{label}_95hdi.csv"
    summary.to_csv(out)
    print(f"  wrote {out}  ({summary.shape[0]} rows)")


print("\nExporting 95% HDI posterior summaries ...")
export_summary(idata_main,  "main")
export_summary(idata_lag2,  "lag2")
export_summary(idata_inter, "inter")


# -----------------------------------------------------------------------------
# 3. Figure 1 — Forest plot of phi, beta_p, beta_s (95% HDI)
# -----------------------------------------------------------------------------
print("\nBuilding Figure 1 (forest plot, 95% HDI) ...")

# Adjust these names if your model used different parameter labels.
candidate_var_names = ["phi", "beta_p", "beta_s"]
posterior_vars = list(idata_main.posterior.data_vars)
var_names = [v for v in candidate_var_names if v in posterior_vars]
if len(var_names) != 3:
    print(f"  WARNING: expected {candidate_var_names}, found {var_names}")
    print(f"  available posterior vars: {posterior_vars}")

pretty_labels = {
    "phi":    r"Inertia ($\varphi$)",
    "beta_p": r"Mortality ($\beta_{p}$)",
    "beta_s": r"Stringency ($\beta_{s}$)",
}

fig1, ax1 = plt.subplots(figsize=(7.0, 3.0), dpi=300)
az.plot_forest(
    idata_main,
    var_names=var_names,
    combined=True,
    hdi_prob=0.95,        # override ArviZ default (0.94)
    colors="black",
    markersize=6,
    linewidth=2.0,
    ax=ax1,
)
ax1.axvline(0.0, color="grey", linestyle="--", linewidth=1.0, zorder=0)
ax1.set_yticklabels([pretty_labels.get(v, v) for v in var_names], fontsize=11)
ax1.set_xlabel("Posterior mean and 95% HDI", fontsize=11)
ax1.set_title("")
ax1.tick_params(axis="x", labelsize=10)
for spine in ("top", "right"):
    ax1.spines[spine].set_visible(False)

fig1.tight_layout()
fig1_pdf = FIGURES_DIR / "fig_bayesian_dynamic_forest.pdf"
fig1.savefig(fig1_pdf, bbox_inches="tight")
fig1.savefig(fig1_pdf.with_suffix(".png"), bbox_inches="tight", dpi=300)
plt.close(fig1)
print(f"  wrote {fig1_pdf}")


# -----------------------------------------------------------------------------
# 4. Figure 2 — Raw bivariate selection-effect scatter
# -----------------------------------------------------------------------------
print("\nBuilding Figure 2 (selection-effect scatter) ...")

# Build country-level dataframe from the long panel:
#   cohesion_2019            = v2smpolsoc in 2019
#   stringency_mean_2020_21  = mean of stringency_mean over 2020-2021
cohesion_2019 = (
    df_panel.loc[df_panel["year"] == 2019, ["iso3", "v2smpolsoc"]]
    .rename(columns={"v2smpolsoc": "cohesion_2019"})
)
stringency_2020_21 = (
    df_panel.loc[df_panel["year"].isin([2020, 2021]), ["iso3", "stringency_mean"]]
    .groupby("iso3", as_index=False)["stringency_mean"].mean()
    .rename(columns={"stringency_mean": "stringency_mean_2020_21"})
)
plot_df = (
    cohesion_2019.merge(stringency_2020_21, on="iso3", how="inner")
    .dropna()
    .reset_index(drop=True)
)
print(f"  N countries with both variables: {len(plot_df)}")

x = plot_df["cohesion_2019"].to_numpy()
y = plot_df["stringency_mean_2020_21"].to_numpy()
r_val, p_val = stats.pearsonr(x, y)
print(f"  Pearson r = {r_val:.3f}, p = {p_val:.4f}")

sns.set_style("white")
fig2, ax2 = plt.subplots(figsize=(7.5, 6.0), dpi=300)

sns.regplot(
    x="cohesion_2019",
    y="stringency_mean_2020_21",
    data=plot_df,
    ci=95,
    scatter_kws={
        "s": 36, "color": "#2b6cb0",
        "edgecolor": "white", "linewidth": 0.8, "alpha": 0.9,
    },
    line_kws={"color": "black", "linewidth": 1.6},
    ax=ax2,
)

for _, row in plot_df.iterrows():
    ax2.annotate(
        row["iso3"],
        xy=(row["cohesion_2019"], row["stringency_mean_2020_21"]),
        xytext=(4, 4), textcoords="offset points",
        fontsize=8, color="black",
    )

ax2.text(
    0.97, 0.97,
    f"$r = {r_val:.2f}$, $p = {p_val:.3f}$",
    transform=ax2.transAxes, ha="right", va="top",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.4",
              facecolor="white", edgecolor="black", linewidth=0.8),
)

ax2.set_xlabel("Pre-pandemic social cohesion (V-Dem v2smpolsoc, 2019)", fontsize=11)
ax2.set_ylabel("Mean lockdown stringency, 2020–2021\n"
               "(OxCGRT Stringency Index, 0–100)", fontsize=11)
ax2.tick_params(axis="both", labelsize=10)
for spine in ("top", "right"):
    ax2.spines[spine].set_visible(False)

fig2.tight_layout()
fig2_pdf = FIGURES_DIR / "fig_selection_effect_scatter.pdf"
fig2.savefig(fig2_pdf, bbox_inches="tight")
fig2.savefig(fig2_pdf.with_suffix(".png"), bbox_inches="tight", dpi=300)
plt.close(fig2)
print(f"  wrote {fig2_pdf}")

print("\nDone.")
