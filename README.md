# The Stress Test That Changed Nothing

**Pre-pandemic Social Cohesion Determined Lockdown Strategy but Neither the Virus nor the Response Drove Polarization in OECD Nations**

Replication materials for the manuscript.

**Repository:** [https://github.com/denovo2021/EIPSA](https://github.com/denovo2021/EIPSA)

---

## Abstract

**Background.** The Parasite-Stress Theory of Values predicts that infectious disease threats drive societies toward authoritarianism and social division. The COVID-19 pandemic offered an unprecedented test of this hypothesis across developed democracies, yet it remains unclear whether the political disruptions observed during the pandemic were caused by the biological threat, the policy response, or neither.

**Methods.** We conducted a panel data analysis of 38 OECD countries (2019–2024) integrating the Varieties of Democracy Project (V-Dem v15) for social polarization indicators, the EM-DAT International Disaster Database for historical epidemic exposure, and the Oxford COVID-19 Government Response Tracker (OxCGRT) via Our World in Data for mortality and lockdown stringency measures. We employed two-way fixed-effects models with first-differenced outcomes and clustered standard errors, supplemented by cross-sectional analyses with robust inference.

**Results.** We report a double null finding. COVID-19 mortality did not predict changes in social polarization (*p* = 0.25). Lockdown stringency did not predict increased polarization either; the coefficient was negative and non-significant (*p* = 0.13). The primary finding is a selection effect: pre-pandemic social cohesion was strongly negatively correlated with subsequent lockdown stringency (*r* = −0.51, *p* = 0.001). Cohesive societies achieved compliance voluntarily and adopted lighter restrictions; polarized societies resorted to stringent mandates.

**Conclusion.** The pandemic served as a stress test that revealed, rather than created, the fractures in OECD democracies. Pandemic preparedness is ultimately a function of social capital accumulated long before a crisis arrives.

---

## System Requirements

- **Python** 3.8 or higher
- Standard scientific computing libraries (see `requirements.txt`)
- Tested on Python 3.9 (macOS) and Python 3.11 (Ubuntu 22.04)
- No GPU or high-memory configuration required
- Expected run time: ~1 minute total on a standard laptop

## Installation

```bash
git clone https://github.com/denovo2021/EIPSA.git
cd EIPSA
pip install -r requirements.txt
```

---

## Repository Structure

```
EIPSA/
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   └── EIPSA_OECD_panel_2019_2024.csv   # Merged analysis-ready dataset
│
├── scripts/
│   ├── 01_main_analysis.py               # Table 1 + Figures 1–2
│   ├── 02_selection_effect.py            # Figure 3 (scatter plot)
│   └── 03_robustness_checks.py           # SI: GDP, economic controls, jackknife
│
└── output/
    ├── figures/
    │   ├── figure1.png                   # OECD cohesion trend (2010–2024)
    │   ├── figure2.png                   # Coefficient plot (horse-race model)
    │   ├── figure3.png                   # Selection effect scatter
    │   └── si_figure_s1.png              # Forest plot (all specifications)
    └── tables/
        ├── table1.txt                    # Main regression table
        └── si_robustness.txt             # Supplementary statistics
```

---

## Usage

Run the scripts in order from the repository root:

```bash
# Step 1: Reproduce Table 1 (4 regression models) and Figures 1–2
python scripts/01_main_analysis.py

# Step 2: Reproduce Figure 3 (selection effect scatter plot)
python scripts/02_selection_effect.py

# Step 3: Reproduce all Supplementary Information analyses
python scripts/03_robustness_checks.py
```

All outputs are saved to `output/figures/` and `output/tables/`.

---

## Data Availability

The merged dataset `data/EIPSA_OECD_panel_2019_2024.csv` is included in this repository for direct replication. It was constructed from the following public sources:

| Dataset | Source | URL |
|---------|--------|-----|
| **V-Dem v15** | Varieties of Democracy Project | https://v-dem.net/data/the-v-dem-dataset/ |
| **OWID COVID-19** | Our World in Data | https://github.com/owid/covid-19-data |
| **OxCGRT** | Oxford COVID-19 Government Response Tracker | https://github.com/OxCGRT/covid-policy-dataset |
| **EM-DAT** | CRED, UCLouvain | https://www.emdat.be/ |

### Key Variables

| Variable | Source | Description |
|----------|--------|-------------|
| `v2smpolsoc` | V-Dem | Social polarization (**high = cohesive**, low = polarized) |
| `v2cacamps` | V-Dem | Political polarization (high = polarized) |
| `v2x_libdem` | V-Dem | Liberal democracy index (0–1) |
| `stringency_norm` | OxCGRT | Lockdown stringency (0–1, normalized) |
| `covid_intensity` | OWID | log(1 + annual deaths per million) |
| `econ_support_norm` | OxCGRT | Economic Support Index (0–1, normalized) |

**Outcome coding:** Polarization = −v2smpolsoc, so positive coefficients indicate increased polarization.

---

## Citation

> [Authors]. (2026). The stress test that changed nothing: Pre-pandemic social cohesion determined lockdown strategy but neither the virus nor the response drove polarization in OECD nations. *Nature Human Behaviour*. [Manuscript submitted for publication].

---

## License

Code: MIT License.
Data: Subject to the respective providers' terms of use (V-Dem, OWID, OxCGRT, EM-DAT).
