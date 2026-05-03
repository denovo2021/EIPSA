"""Export Supplementary Tables S1 and S2 at the strict 95% HDI reporting
standard, using the traces produced by ``fit_sensitivity_lag2.py`` and
``fit_sensitivity_interaction.py``.

Outputs
-------
    output/tables/posterior_summary_si_lag2_95hdi.csv         (Supp. Table S1)
    output/tables/posterior_summary_si_interaction_95hdi.csv  (Supp. Table S2)
"""
from __future__ import annotations
from pathlib import Path

import arviz as az

ROOT   = Path(__file__).resolve().parent.parent
OUT    = ROOT / "output"
TABLES = OUT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def export(idata_path: Path, var_names: list[str], out_csv: Path) -> None:
    idata   = az.from_netcdf(idata_path)
    summary = az.summary(idata, var_names=var_names, hdi_prob=0.95)
    summary.to_csv(out_csv, index=True)
    print(f"[saved] {out_csv}")


def main() -> None:
    # Supplementary Table S1 - Lag-2 sensitivity
    export(
        idata_path=OUT / "idata_sensitivity_lag2.nc",
        var_names=["phi", "beta_p", "beta_s", "gamma",
                   "mu_a", "sigma_region", "sigma_country", "sigma_y"],
        out_csv=TABLES / "posterior_summary_si_lag2_95hdi.csv",
    )
    # Supplementary Table S2 - Effect-modification (interaction)
    export(
        idata_path=OUT / "idata_sensitivity_interaction.nc",
        var_names=["phi", "beta_p", "beta_s", "delta_p", "delta_s", "gamma",
                   "mu_a", "sigma_region", "sigma_country", "sigma_y"],
        out_csv=TABLES / "posterior_summary_si_interaction_95hdi.csv",
    )


if __name__ == "__main__":
    main()
