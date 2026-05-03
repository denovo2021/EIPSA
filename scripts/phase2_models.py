"""
Phase 2: panel and cross-sectional models for the EIPSA revision.

Primary:    TWFE-on-levels with Driscoll-Kraay SE.
Secondary:  cluster-by-country, heteroskedastic-robust, region-WCB.
Robustness: First-Difference; spatial-lag (W*y) cross-sectional regressor.
Cross-sec:  stepwise multiple OLS replacing the FWL decomposition.

Run from project root, after phase1_data.py:
    python scripts/phase2_models.py

Outputs: output/tables/*.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, FirstDifferenceOLS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TABLES = ROOT / "output" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

OUTCOME      = "v2smpolsoc"
EXPOSURES    = ["p_score_mean", "stringency_mean"]
TIME_VARYING = ["log_pop", "urban_pct", "health_exp_gdp", "log_gdp_pc"]


def load_panel() -> pd.DataFrame:
    return pd.read_parquet(DATA / "oecd_panel.parquet")


def _prep_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_pop"]    = np.log(df["population"])
    df["log_gdp_pc"] = np.log(df["gdp_pc_ppp"])
    df = df.dropna(subset=[OUTCOME, *EXPOSURES, *TIME_VARYING])
    return df.set_index(["iso3", "year"])


def _fmt(res, label: str) -> pd.DataFrame:
    p  = res.params
    se = res.std_errors if hasattr(res, "std_errors") else res.bse
    ci = res.conf_int() if hasattr(res, "conf_int") else None
    out = pd.DataFrame({"coef": p, "se": se, "t": p / se})
    if ci is not None:
        ci = ci.copy()
        ci.columns = ["ci_low", "ci_high"]
        out = out.join(ci)
    out["spec"] = label
    return out.reset_index().rename(columns={"index": "term"})


# -- Primary: TWFE on levels -------------------------------------------------
def twfe_panel(df: pd.DataFrame) -> pd.DataFrame:
    d = _prep_panel(df)
    rhs = EXPOSURES + TIME_VARYING
    X = sm.add_constant(d[rhs])
    mod = PanelOLS(d[OUTCOME], X, entity_effects=True, time_effects=True,
                   drop_absorbed=True)
    res_dk      = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=2)
    res_cluster = mod.fit(cov_type="clustered", cluster_entity=True)
    res_robust  = mod.fit(cov_type="robust")
    return pd.concat([
        _fmt(res_dk,      "TWFE / Driscoll-Kraay"),
        _fmt(res_cluster, "TWFE / cluster-by-country"),
        _fmt(res_robust,  "TWFE / heteroskedastic-robust"),
    ], ignore_index=True)


# -- Region-WCB SE (8 clusters, small-cluster robust) -----------------------
def twfe_region_wcb(df: pd.DataFrame, B: int = 1999, seed: int = 7) -> pd.DataFrame:
    """Cameron-Gelbach-Miller wild-cluster bootstrap-t at UN sub-region."""
    d = _prep_panel(df)
    region_full = df.set_index(["iso3", "year"])["region"]
    region = region_full.loc[d.index].values
    rhs = EXPOSURES + TIME_VARYING
    X = sm.add_constant(d[rhs])
    base = PanelOLS(d[OUTCOME], X, entity_effects=True, time_effects=True,
                    drop_absorbed=True).fit(cov_type="unadjusted")
    rng = np.random.default_rng(seed)
    fitted = base.fitted_values.values.flatten()
    resid  = base.resids.values
    regions = np.unique(region)
    boots = {k: [] for k in EXPOSURES}
    for _ in range(B):
        flips = rng.choice([-1.0, 1.0], size=regions.size)
        flip_map = pd.Series(flips, index=regions)
        weight = flip_map.reindex(region).values
        d_star = d.copy()
        d_star[OUTCOME] = fitted + resid * weight
        res = PanelOLS(d_star[OUTCOME], X, entity_effects=True, time_effects=True,
                       drop_absorbed=True).fit(cov_type="unadjusted")
        for k in EXPOSURES:
            boots[k].append(res.params[k])
    rows = []
    for k in EXPOSURES:
        b = np.array(boots[k])
        rows.append({"term": k,
                     "coef": base.params[k],
                     "se": b.std(ddof=1),
                     "ci_low":  np.quantile(b, 0.025),
                     "ci_high": np.quantile(b, 0.975),
                     "spec": "TWFE / region-WCB"})
    return pd.DataFrame(rows)


# -- Robustness: First-Difference -------------------------------------------
def first_difference(df: pd.DataFrame) -> pd.DataFrame:
    """Manual within-country first differences, then OLS with country-clustered
    SE. Avoids linearmodels' rank-based constant detection in FirstDifferenceOLS,
    which can trip on multicollinearity even when no explicit intercept column
    is supplied. Mathematically equivalent to a no-intercept FD regression."""
    d = _prep_panel(df).reset_index().sort_values(["iso3", "year"])
    rhs = EXPOSURES + TIME_VARYING
    diffs = (d.groupby("iso3")[[OUTCOME, *rhs]]
               .diff()
               .dropna())
    iso = d.loc[diffs.index, "iso3"].values
    y = diffs[OUTCOME]
    X = diffs[rhs]
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": iso})
    ci = res.conf_int()
    out = pd.DataFrame({"coef": res.params, "se": res.bse, "t": res.tvalues,
                        "ci_low": ci[0], "ci_high": ci[1]})
    out["spec"] = "FD / cluster-by-country"
    return out.reset_index().rename(columns={"index": "term"})


# -- Robustness: spatial-lag W*y on country-mean cross-section --------------
def spatial_lag_check(df: pd.DataFrame) -> pd.DataFrame:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from libpysal.weights import KNN
    from phase1_data import CAPITAL_COORDS  # type: ignore  # noqa
    cs = (df.groupby("iso3", as_index=False)
            .agg({OUTCOME: "mean",
                  "p_score_mean": "mean",
                  "stringency_mean": "mean",
                  "population": "mean",
                  "urban_pct": "mean",
                  "health_exp_gdp": "mean",
                  "gdp_pc_ppp": "mean"}))
    cs["log_pop"]    = np.log(cs["population"])
    cs["log_gdp_pc"] = np.log(cs["gdp_pc_ppp"])
    cs["lat"] = cs["iso3"].map(lambda c: CAPITAL_COORDS[c][0])
    cs["lon"] = cs["iso3"].map(lambda c: CAPITAL_COORDS[c][1])
    cs = cs.dropna(subset=[OUTCOME, "p_score_mean", "stringency_mean",
                            "log_pop", "urban_pct", "health_exp_gdp", "log_gdp_pc"])
    pts = cs[["lon", "lat"]].values
    w = KNN.from_array(pts, k=4)
    w.transform = "r"
    cs["Wy"] = w.sparse @ cs[OUTCOME].values
    rhs = ["p_score_mean", "stringency_mean", "log_pop", "urban_pct",
           "health_exp_gdp", "log_gdp_pc", "Wy"]
    X = sm.add_constant(cs[rhs])
    res = sm.OLS(cs[OUTCOME], X).fit(cov_type="HC3")
    ci = res.conf_int()
    out = pd.DataFrame({"coef": res.params, "se": res.bse, "t": res.tvalues,
                        "p": res.pvalues,
                        "ci_low": ci[0], "ci_high": ci[1]})
    out["spec"] = "Cross-section + W*y (HC3)"
    return out.reset_index().rename(columns={"index": "term"})


# -- Cross-section: association of 2019 cohesion with 2020-21 mean stringency
CROSS_SPECS = {
    "M1: bivariate":      ["cohesion_2019"],
    "M2: + demographics": ["cohesion_2019", "log_pop", "urban_pct", "pop_density"],
    "M3: + economy":      ["cohesion_2019", "log_pop", "urban_pct", "pop_density",
                           "log_gdp_pc", "health_exp_gdp"],
    "M4: + ethnic frac":  ["cohesion_2019", "log_pop", "urban_pct", "pop_density",
                           "log_gdp_pc", "health_exp_gdp", "ethnic_frac"],
    "M5: + region FE":    ["cohesion_2019", "log_pop", "urban_pct", "pop_density",
                           "log_gdp_pc", "health_exp_gdp", "ethnic_frac"],
}


def build_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    pre = (df[df.year == 2019]
             .set_index("iso3")[[OUTCOME, "ethnic_frac", "population",
                                 "urban_pct", "health_exp_gdp", "gdp_pc_ppp",
                                 "pop_density", "region"]]
             .rename(columns={OUTCOME: "cohesion_2019"}))
    pre["log_pop"]    = np.log(pre["population"])
    pre["log_gdp_pc"] = np.log(pre["gdp_pc_ppp"])
    str_2020_21 = (df[df.year.isin([2020, 2021])]
                     .groupby("iso3")["stringency_mean"].mean()
                     .rename("stringency_2020_21"))
    cs = pre.join(str_2020_21).dropna(subset=["stringency_2020_21", "cohesion_2019"])
    return cs


def cross_section_models(df: pd.DataFrame) -> pd.DataFrame:
    cs = build_cross_section(df)
    rows = []
    for label, rhs in CROSS_SPECS.items():
        X = cs[rhs].copy()
        if label == "M5: + region FE":
            X = pd.concat([X, pd.get_dummies(cs["region"], prefix="reg",
                                              drop_first=True, dtype=float)], axis=1)
        X = sm.add_constant(X)
        X = X.dropna()
        y = cs.loc[X.index, "stringency_2020_21"]
        res = sm.OLS(y, X).fit(cov_type="HC3")
        ci = res.conf_int()
        for term in res.params.index:
            rows.append({"spec": label, "term": term,
                         "coef": res.params[term], "se": res.bse[term],
                         "t": res.tvalues[term], "p": res.pvalues[term],
                         "ci_low": ci.loc[term, 0], "ci_high": ci.loc[term, 1],
                         "n": int(res.nobs), "r2": res.rsquared,
                         "r2_adj": res.rsquared_adj})
    return pd.DataFrame(rows)


def fit_m5(df: pd.DataFrame):
    """Return (model_results, design_X, y, cs_df) for the AVP in Phase 3."""
    cs = build_cross_section(df)
    rhs = CROSS_SPECS["M5: + region FE"]
    X = cs[rhs].copy()
    X = pd.concat([X, pd.get_dummies(cs["region"], prefix="reg",
                                      drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X).dropna()
    y = cs.loc[X.index, "stringency_2020_21"]
    res = sm.OLS(y, X).fit(cov_type="HC3")
    return res, X, y, cs.loc[X.index]


# -- Run all -----------------------------------------------------------------
def main():
    panel = load_panel()
    panel_main = twfe_panel(panel)
    panel_wcb  = twfe_region_wcb(panel, B=1999)
    panel_fd   = first_difference(panel)
    sp_lag     = spatial_lag_check(panel)
    cross      = cross_section_models(panel)

    panel_main.to_csv(TABLES / "panel_twfe.csv",        index=False)
    panel_wcb .to_csv(TABLES / "panel_twfe_wcb.csv",    index=False)
    panel_fd  .to_csv(TABLES / "panel_fd.csv",          index=False)
    sp_lag    .to_csv(TABLES / "spatial_lag_check.csv", index=False)
    cross     .to_csv(TABLES / "cross_section.csv",     index=False)

    forest = pd.concat([panel_main, panel_wcb, panel_fd], ignore_index=True)
    forest = forest[forest.term.isin(EXPOSURES)].copy()
    forest.to_csv(TABLES / "forest_input.csv", index=False)

    print("=== Primary panel (TWFE) — exposures only ===")
    print(panel_main[panel_main.term.isin(EXPOSURES)].to_string(index=False))
    print("\n=== Cross-section: cohesion_2019 row across specs ===")
    print(cross[cross.term == "cohesion_2019"].to_string(index=False))


if __name__ == "__main__":
    main()
