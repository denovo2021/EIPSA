"""
Phase 3: academic-grade figures for the EIPSA revision.

Figures produced:
    output/figures/fig_avp_cohesion_stringency.{pdf,png}
        Added-variable plot for cross-sectional M5: cohesion_2019 vs.
        stringency_2020-21, partialling out demographics, economy, ethnic
        fractionalization, and UN sub-region fixed effects.

    output/figures/fig_forest_panel.{pdf,png}
        Forest plot of the panel (TWFE) coefficients describing the
        association of excess mortality (P-score) and lockdown stringency
        with social polarization, across the four primary SE specifications
        and the FD robustness check.

Run from project root, after phase2_models.py:
    python scripts/phase3_plots.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_models import (  # noqa: E402
    load_panel, fit_m5, twfe_panel, twfe_region_wcb, first_difference,
    EXPOSURES,
)

FIGS   = ROOT / "output" / "figures"
TABLES = ROOT / "output" / "tables"
FIGS.mkdir(parents=True, exist_ok=True)

# --- Journal-grade matplotlib style ----------------------------------------
mpl.rcParams.update({
    "figure.dpi":          300,
    "savefig.dpi":         600,
    "savefig.bbox":        "tight",
    "savefig.transparent": False,
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":           9,
    "axes.titlesize":      10,
    "axes.labelsize":      9,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     8,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.linewidth":      0.8,
    "xtick.major.width":   0.8,
    "ytick.major.width":   0.8,
    "lines.linewidth":     1.2,
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
})

NAVY   = "#1f3a5f"
ORANGE = "#d97706"
GREY   = "#666666"


# ---------------------------------------------------------------------------
# (a) Added-variable plot for cross-sectional M5
# ---------------------------------------------------------------------------
def added_variable_plot(panel: pd.DataFrame) -> None:
    res_m5, X, y, cs = fit_m5(panel)

    focal = "cohesion_2019"
    other = [c for c in X.columns if c not in (focal, "const")]
    Xo = sm.add_constant(X[other])

    y_resid     = y      - sm.OLS(y,        Xo).fit().fittedvalues
    focal_resid = X[focal] - sm.OLS(X[focal], Xo).fit().fittedvalues

    avp_X = sm.add_constant(focal_resid.rename("focal_resid"))
    avp   = sm.OLS(y_resid, avp_X).fit(cov_type="HC3")
    slope = avp.params["focal_resid"]
    se    = avp.bse["focal_resid"]
    pval  = avp.pvalues["focal_resid"]
    ci_lo, ci_hi = avp.conf_int().loc["focal_resid"].tolist()

    xs = np.linspace(focal_resid.min() - 0.05, focal_resid.max() + 0.05, 200)
    Xs = sm.add_constant(pd.Series(xs, name="focal_resid"))
    pred = avp.get_prediction(Xs).summary_frame(alpha=0.05)

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.fill_between(xs, pred["mean_ci_lower"], pred["mean_ci_upper"],
                    color=NAVY, alpha=0.12, linewidth=0)
    ax.plot(xs, pred["mean"], color=NAVY, linewidth=1.4)
    ax.scatter(focal_resid, y_resid, s=22, facecolor="white",
               edgecolor=NAVY, linewidth=0.9, zorder=3)

    rmag = np.hypot(focal_resid / focal_resid.std(),
                    y_resid     / y_resid.std())
    label_idx = rmag.sort_values(ascending=False).index[:14]
    for iso in label_idx:
        ax.annotate(iso, (focal_resid.loc[iso], y_resid.loc[iso]),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=7, color=GREY)

    ax.axhline(0, color=GREY, linewidth=0.5, linestyle=":")
    ax.axvline(0, color=GREY, linewidth=0.5, linestyle=":")
    ax.set_xlabel("Pre-pandemic social cohesion (2019), residualized")
    ax.set_ylabel("Mean lockdown stringency (2020–21), residualized")
    ax.set_title("Added-variable plot — M5 partial association",
                 loc="left", pad=8)

    txt = (f"Partial slope: {slope:.2f}  (HC3 SE {se:.2f})\n"
           f"95 % CI [{ci_lo:.2f}, {ci_hi:.2f}]   p = {pval:.3f}\n"
           f"n = {int(avp.nobs)}   adj. R² of M5 = {res_m5.rsquared_adj:.2f}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5, color="black",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=GREY, linewidth=0.5))

    out_pdf = FIGS / "fig_avp_cohesion_stringency.pdf"
    out_png = FIGS / "fig_avp_cohesion_stringency.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[avp] wrote {out_pdf.name} & {out_png.name}  "
          f"(slope={slope:.3f}, p={pval:.3f}, n={int(avp.nobs)})")


# ---------------------------------------------------------------------------
# (b) Coefficient forest plot
# ---------------------------------------------------------------------------
SPEC_ORDER = [
    "TWFE / Driscoll-Kraay",
    "TWFE / cluster-by-country",
    "TWFE / heteroskedastic-robust",
    "TWFE / region-WCB",
    "FD / cluster-by-country",
]
EXPOSURE_LABEL = {
    "p_score_mean":    "Excess mortality (P-score)",
    "stringency_mean": "Lockdown stringency",
}


def _ensure_forest_input(panel: pd.DataFrame) -> pd.DataFrame:
    fp = TABLES / "forest_input.csv"
    if fp.exists():
        return pd.read_csv(fp)
    df = pd.concat([twfe_panel(panel),
                    twfe_region_wcb(panel, B=1999),
                    first_difference(panel)], ignore_index=True)
    df = df[df.term.isin(EXPOSURES)].copy()
    df.to_csv(fp, index=False)
    return df


def forest_plot(panel: pd.DataFrame) -> None:
    df = _ensure_forest_input(panel)
    df = df[df.spec.isin(SPEC_ORDER) & df.term.isin(EXPOSURES)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, term in zip(axes, EXPOSURES):
        sub = (df[df.term == term]
                 .set_index("spec").reindex(SPEC_ORDER).reset_index())
        y_pos = np.arange(len(sub))[::-1]
        ax.errorbar(sub["coef"], y_pos,
                    xerr=[sub["coef"] - sub["ci_low"],
                          sub["ci_high"] - sub["coef"]],
                    fmt="o", color=NAVY, ecolor=NAVY,
                    markersize=4.5, capsize=2.5, linewidth=1.0)
        ax.axvline(0, color=GREY, linewidth=0.6, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["spec"])
        ax.set_xlabel("Coefficient (association with social polarization, v2smpolsoc)")
        ax.set_title(EXPOSURE_LABEL[term], loc="left", pad=6)

        xmax = sub["ci_high"].max()
        xmin = sub["ci_low"].min()
        pad  = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
        for yi, (_, row) in zip(y_pos, sub.iterrows()):
            ax.text(xmax + pad, yi,
                    f"{row['coef']:.3f}  [{row['ci_low']:.3f}, {row['ci_high']:.3f}]",
                    va="center", fontsize=7, color="black")
        ax.set_xlim(xmin - pad, xmax + 5 * pad)

    fig.suptitle("Panel TWFE coefficients across SE specifications and FD robustness",
                 fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_pdf = FIGS / "fig_forest_panel.pdf"
    out_png = FIGS / "fig_forest_panel.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[forest] wrote {out_pdf.name} & {out_png.name}")


# ---------------------------------------------------------------------------
def main():
    panel = load_panel()
    added_variable_plot(panel)
    forest_plot(panel)


if __name__ == "__main__":
    main()
