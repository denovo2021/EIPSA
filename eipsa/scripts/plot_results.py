"""Posterior summary tables and figures for the EIPSA manuscript.

Generates from one or two fitted :class:`arviz.InferenceData` objects:

* ``Table 1``  — 95% HDI posterior summary of the Lag-1 model.
* ``Figure 1`` — forest plot of the three coefficients of association
  (phi, beta_p, beta_s) at the 95% HDI.
* ``Figure 2`` — country-level bivariate association between pre-pandemic
  social cohesion (V-Dem v2smpolsoc, 2019) and the mean OxCGRT stringency
  index over 2020–2021.
* ``Supplementary Table S1`` — 95% HDI posterior summary of the Lag-2
  sensitivity model.
* ``Supplementary Table S2`` — convergence diagnostics (R-hat, ESS) for
  both fitted models, aggregated across all monitored parameters.

Türkiye (TUR) is excluded from the Figure 2 cross-section because the
lagged log COVID-mortality regressor required by the dynamic model is
not reported for Türkiye over 2019–2024; the exclusion keeps the
cross-section consistent with the Bayesian analytic sample (N = 37).
"""
from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .data_prep import PANEL_PATH

# Parameters reported in Table 1 / Supplementary Table S1.
SUMMARY_VARS = [
    "phi", "beta_p", "beta_s", "gamma",
    "mu_a", "sigma_region", "sigma_country", "sigma_y",
]

EXCLUDE_ISO3 = ["TUR"]  # see module docstring

PRETTY = {
    "phi":    r"Inertia ($\varphi$)",
    "beta_p": r"Mortality ($\beta_p$)",
    "beta_s": r"Stringency ($\beta_s$)",
}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def export_posterior_summary(
    idata: az.InferenceData,
    out_csv: Path,
    var_names: list[str] = SUMMARY_VARS,
) -> Path:
    """Write a 95% HDI posterior summary table to CSV."""
    summary = az.summary(idata, var_names=var_names, hdi_prob=0.95)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=True)
    print(f"[saved] {out_csv}")
    return out_csv


def export_convergence_diagnostics(
    idatas: dict[str, az.InferenceData],
    out_csv: Path,
    var_names: list[str] = SUMMARY_VARS,
) -> Path:
    """Aggregate R-hat / ESS diagnostics across all fitted models."""
    rows = []
    for label, idata in idatas.items():
        s = az.summary(idata, var_names=var_names, hdi_prob=0.95)
        rows.append(
            {
                "model": label,
                "n_parameters": int(s.shape[0]),
                "max_r_hat": float(s["r_hat"].max()),
                "mean_r_hat": float(s["r_hat"].mean()),
                "min_ess_bulk": float(s["ess_bulk"].min()),
                "min_ess_tail": float(s["ess_tail"].min()),
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")
    return out_csv


# ---------------------------------------------------------------------------
# Figure 1 — forest plot
# ---------------------------------------------------------------------------
def figure1_forest(idata: az.InferenceData, out_pdf: Path) -> Path:
    """Forest plot of phi, beta_p, beta_s at the 95% HDI."""
    var_names = ["phi", "beta_p", "beta_s"]
    axes = az.plot_forest(
        idata,
        var_names=var_names,
        combined=True,
        hdi_prob=0.95,
        figsize=(7.0, 2.8),
    )
    ax = axes[0] if hasattr(axes, "__iter__") else axes
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")

    new_labels = []
    for lbl in ax.get_yticklabels():
        txt = lbl.get_text().strip()
        new_labels.append(PRETTY.get(txt, txt))
    ax.set_yticklabels(new_labels)
    ax.set_title("Posterior coefficients of association (95% HDI)", fontsize=11)

    fig = ax.figure
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[saved] {out_pdf}")
    return out_pdf


# ---------------------------------------------------------------------------
# Figure 2 — bivariate selection-effect scatter
# ---------------------------------------------------------------------------
def figure2_selection_effect(
    out_pdf: Path,
    panel_path: Path = PANEL_PATH,
    out_stats_csv: Path | None = None,
) -> Path:
    """Country-level association between baseline cohesion and stringency.

    The reported Pearson coefficient quantifies the strength of the
    bivariate association on the analytic OECD sample (N = 37); it
    complements — and does not replace — the Bayesian estimate of the
    within-country temporal association reported in Figure 1.
    """
    df = pd.read_csv(panel_path)
    df = df[~df["iso3"].isin(EXCLUDE_ISO3)].copy()

    cohesion_2019 = (
        df.loc[df["year"] == 2019, ["iso3", "v2smpolsoc"]]
          .rename(columns={"v2smpolsoc": "cohesion_2019"})
    )
    stringency_2020_21 = (
        df.loc[df["year"].isin([2020, 2021])]
          .groupby("iso3", as_index=False)["stringency_mean"].mean()
          .rename(columns={"stringency_mean": "stringency_mean_2020_21"})
    )
    plot_df = cohesion_2019.merge(stringency_2020_21, on="iso3").dropna()

    r, p = stats.pearsonr(
        plot_df["cohesion_2019"], plot_df["stringency_mean_2020_21"]
    )

    if out_stats_csv is not None:
        out_stats_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "n": len(plot_df),
                    "excluded": ",".join(EXCLUDE_ISO3) or "none",
                    "pearson_r": r,
                    "p_value": p,
                }
            ]
        ).to_csv(out_stats_csv, index=False)
        print(f"[saved] {out_stats_csv}")

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=300)
    sns.regplot(
        data=plot_df,
        x="cohesion_2019",
        y="stringency_mean_2020_21",
        ax=ax,
        ci=95,
        scatter_kws={
            "s": 42, "alpha": 0.85,
            "edgecolor": "white", "linewidths": 0.6,
        },
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
        f"$r = {r:.2f},\\ p = {p:.3f},\\ N = {len(plot_df)}$",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor="white", edgecolor="0.7", alpha=0.9),
    )
    ax.set_xlabel("Pre-pandemic social cohesion (V-Dem v2smpolsoc, 2019)")
    ax.set_ylabel("Mean lockdown stringency, 2020–2021 (OxCGRT, 0–100)")
    ax.set_title(
        "Bivariate association: baseline cohesion and subsequent stringency"
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(
        f"[saved] {out_pdf}  "
        f"(Pearson r = {r:.3f}, p = {p:.3f}, N = {len(plot_df)})"
    )
    return out_pdf
