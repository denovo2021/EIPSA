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

idata_main  = az.from_netcdf(IDATA_MAIN)
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
# 3. Figure 1 - Forest plot (95% HDI)
# -----------------------------------------------------------------------------
# The saved idata uses region-varying coefficients beta_p[region], beta_s[region]
# rather than the scalar phi/beta_p/beta_s described in the manuscript text.
# We pool across the region dimension and plot one row per parameter.
print("\nBuilding Figure 1 (forest plot, 95% HDI) ...")

post = idata_main.posterior
print(f"  posterior variables: {sorted(post.data_vars)}")

import numpy as np
import xarray as xr

def _hdi(arr_1d: np.ndarray, prob: float = 0.95) -> tuple[float, float]:
    """Two-sided HDI of a flat 1-D sample."""
    sorted_arr = np.sort(arr_1d)
    n = len(sorted_arr)
    interval_idx = int(np.floor(prob * n))
    n_intervals = n - interval_idx
    interval_widths = sorted_arr[interval_idx:] - sorted_arr[:n_intervals]
    min_idx = int(np.argmin(interval_widths))
    return float(sorted_arr[min_idx]), float(sorted_arr[min_idx + interval_idx])


def _pooled_summary(da: xr.DataArray) -> dict:
    """Pool a (chain, draw, ...) DataArray across all non-sample dims."""
    flat = da.stack(_sample=("chain", "draw")).values.reshape(-1)
    lo, hi = _hdi(flat, 0.95)
    return {"mean": float(flat.mean()), "lo": lo, "hi": hi}


# Build the three rows for the plot, with graceful fallbacks.
rows = []  # list of (label, mean, lo, hi)

# Row 1: AR / inertia parameter (phi). May not exist in this idata.
phi_candidates = [v for v in ("phi", "phi_ar", "rho", "ar1") if v in post.data_vars]
if phi_candidates:
    s = _pooled_summary(post[phi_candidates[0]])
    rows.append((r"Inertia ($\varphi$)", s["mean"], s["lo"], s["hi"]))
else:
    print("  NOTE: no scalar AR parameter (phi/rho/ar1) found in idata; "
          "Figure 1 will only show beta_p and beta_s.")

# Row 2: Mortality coefficient.
if "beta_p" in post.data_vars:
    s = _pooled_summary(post["beta_p"])
    rows.append((r"Mortality ($\beta_{p}$, pooled)", s["mean"], s["lo"], s["hi"]))

# Row 3: Stringency coefficient.
if "beta_s" in post.data_vars:
    s = _pooled_summary(post["beta_s"])
    rows.append((r"Stringency ($\beta_{s}$, pooled)", s["mean"], s["lo"], s["hi"]))

if not rows:
    sys.exit("ERROR: no plottable parameters found in idata_main.")

print("  Figure 1 rows (pooled across non-sample dims):")
for lbl, m, lo, hi in rows:
    print(f"    {lbl:<35}  mean={m:+.3f}   95% HDI=[{lo:+.3f}, {hi:+.3f}]")

# Manual forest plot (one row per parameter, 95% HDI bars).
fig1, ax1 = plt.subplots(figsize=(7.0, max(2.0, 0.6 * len(rows) + 1.2)), dpi=300)
y_positions = list(range(len(rows)))[::-1]  # top row drawn first
for y, (label, m, lo, hi) in zip(y_positions, rows):
    ax1.errorbar(
        m, y,
        xerr=[[m - lo], [hi - m]],
        fmt="o", color="black",
        markersize=6, linewidth=2.0, capsize=0,
    )

ax1.axvline(0.0, color="grey", linestyle="--", linewidth=1.0, zorder=0)
ax1.set_yticks(y_positions)
ax1.set_yticklabels([r[0] for r in rows], fontsize=11)
ax1.set_ylim(-0.6, len(rows) - 0.4)
ax1.set_xlabel("Posterior mean and 95% HDI", fontsize=11)
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
# 4. Figure 2 - Raw bivariate selection-effect scatter
# -----------------------------------------------------------------------------
print("\nBuilding Figure 2 (selection-effect scatter) ...")

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
        "edgecolor": "white", "linewidths": 0.8, "alpha": 0.9,
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
ax2.set_ylabel("Mean lockdown stringency, 2020-2021\n"
               "(OxCGRT Stringency Index, 0-100)", fontsize=11)
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
