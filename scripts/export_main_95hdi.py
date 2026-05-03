"""Export the posterior-summary table and Figures 1-2 at the strict 95% HDI
reporting standard, using the trace produced by ``fit_main_model_correct.py``.

Outputs
-------
    output/tables/posterior_summary_main_95hdi.csv
    output/figures/fig1_forest_main_95hdi.pdf
    output/figures/fig2_selection_effect_scatter.pdf
"""
from __future__ import annotations
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
OUT     = ROOT / "output"
TABLES  = OUT / "tables"
FIGS    = OUT / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

IDATA_PATH = OUT / "idata_main_correct.nc"

# Türkiye (TUR) is excluded from the analytic panel because cumulative COVID-19
# mortality (Our World in Data; ln(1+deaths per million)) is not reported for
# Türkiye over 2019-2024, so the lagged mortality exposure required by the
# dynamic model cannot be constructed. To keep Figure 2 consistent with the
# Bayesian model's N=37 sample, we exclude TUR from the cross-section here.
EXCLUDE_ISO3 = ["TUR"]


# ---------------------------------------------------------------------------
# Table: posterior summary at 95% HDI (overrides ArviZ default 94%)
# ---------------------------------------------------------------------------
def export_summary_table(idata: az.InferenceData) -> Path:
    summary = az.summary(
        idata,
        var_names=["phi", "beta_p", "beta_s", "gamma",
                   "mu_a", "sigma_region", "sigma_country", "sigma_y"],
        hdi_prob=0.95,
    )
    out_csv = TABLES / "posterior_summary_main_95hdi.csv"
    summary.to_csv(out_csv, index=True)
    print(f"[saved] {out_csv}")
    return out_csv


# ---------------------------------------------------------------------------
# Figure 1: Forest plot at 95% HDI for phi, beta_p, beta_s
# ---------------------------------------------------------------------------
def figure1_forest(idata: az.InferenceData) -> Path:
    var_names = ["phi", "beta_p", "beta_s"]
    pretty    = {"phi":    "Inertia ($\\varphi$)",
                 "beta_p": "Mortality ($\\beta_p$)",
                 "beta_s": "Stringency ($\\beta_s$)"}

    axes = az.plot_forest(
        idata,
        var_names=var_names,
        combined=True,
        hdi_prob=0.95,
        figsize=(7.0, 2.8),
    )
    ax = axes[0] if hasattr(axes, "__iter__") else axes
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")

    # Replace ArviZ tick labels with the requested clean labels.
    new_labels = []
    for lbl in ax.get_yticklabels():
        txt = lbl.get_text().strip()
        for k, v in pretty.items():
            if txt == k:
                new_labels.append(v)
                break
        else:
            new_labels.append(txt)
    ax.set_yticklabels(new_labels)

    ax.set_title("Forest plot of posterior associations (95% HDI)", fontsize=11)
    fig = ax.figure
    fig.tight_layout()
    out_pdf = FIGS / "fig1_forest_main_95hdi.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_pdf}")
    return out_pdf


# ---------------------------------------------------------------------------
# Figure 2: Raw Pearson scatter (cohesion 2019 vs stringency 2020-21 mean)
# ---------------------------------------------------------------------------
def figure2_selection_effect() -> Path:
    df = pd.read_parquet(DATA / "oecd_panel.parquet")
    df = df[~df["iso3"].isin(EXCLUDE_ISO3)].copy()

    cohesion_2019 = (df.loc[df["year"] == 2019, ["iso3", "v2smpolsoc"]]
                       .rename(columns={"v2smpolsoc": "cohesion_2019"}))
    stringency_2020_21 = (
        df.loc[df["year"].isin([2020, 2021])]
          .groupby("iso3", as_index=False)["stringency_mean"].mean()
          .rename(columns={"stringency_mean": "stringency_mean_2020_21"})
    )
    plot_df = cohesion_2019.merge(stringency_2020_21, on="iso3").dropna()

    r, p = stats.pearsonr(plot_df["cohesion_2019"],
                          plot_df["stringency_mean_2020_21"])

    # Persist the recomputed N=37 Pearson statistic for the manuscript text.
    pd.DataFrame(
        [{"n": len(plot_df),
          "excluded": ",".join(EXCLUDE_ISO3) or "none",
          "pearson_r": r,
          "p_value":   p}]
    ).to_csv(TABLES / "selection_effect_pearson_n37.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    sns.regplot(
        data=plot_df,
        x="cohesion_2019",
        y="stringency_mean_2020_21",
        ax=ax,
        ci=95,
        scatter_kws={"s": 42, "alpha": 0.85,
                     "edgecolor": "white", "linewidths": 0.6},
        line_kws={"color": "C3", "linewidth": 1.6},
    )
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["iso3"],
            (row["cohesion_2019"], row["stringency_mean_2020_21"]),
            xytext=(4, 3), textcoords="offset points",
            fontsize=8, color="0.25",
        )

    ax.text(
        0.04, 0.06,
        f"$r = {r:.2f}, p = {p:.3f}$",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor="white", edgecolor="0.7", alpha=0.9),
    )
    ax.set_xlabel("Pre-pandemic social cohesion (V-Dem v2smpolsoc, 2019)")
    ax.set_ylabel("Mean lockdown stringency, 2020-2021 (OxCGRT, 0-100)")
    ax.set_title("Selection effect: pre-pandemic cohesion and subsequent lockdown stringency")

    fig.tight_layout()
    out_pdf = FIGS / "fig2_selection_effect_scatter.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_pdf}  "
          f"(Pearson r={r:.3f}, p={p:.3f}, N={len(plot_df)}, "
          f"excluded={EXCLUDE_ISO3})")
    return out_pdf


def main() -> None:
    idata = az.from_netcdf(IDATA_PATH)
    export_summary_table(idata)
    figure1_forest(idata)
    figure2_selection_effect()


if __name__ == "__main__":
    main()
