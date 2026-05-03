# Supplementary Information

**The Stress Test That Changed Nothing: Pre-pandemic Social Cohesion Determined Lockdown Strategy but Neither the Virus nor the Response Drove Polarization in OECD Nations**

---

## SI Text 1: Robustness to Economic Support

A plausible alternative explanation for the null stringency–polarization finding is that governments simultaneously deployed economic support measures — income subsidies, debt relief, and fiscal transfers — that offset the social costs of lockdowns. Under this "economic buffer" hypothesis, strict lockdowns might indeed erode social cohesion, but the effect would be masked by compensatory fiscal policy.

To test this hypothesis, we augment the main TWFE first-difference model with the OxCGRT Economic Support Index (0–100), which captures income support (E1) and debt/contract relief for households (E2). We estimate:

$$\Delta \text{Polarization}_{it} = \beta_1 \text{Stringency}_{it} + \beta_2 \text{EconSupport}_{it} + \beta_3 (\text{Stringency} \times \text{EconSupport})_{it} + \alpha_i + \gamma_t + \varepsilon_{it}$$

**Results.** In the additive specification, the stringency coefficient remains negative and non-significant ($\hat{\beta}_1$ = −0.913, *p* = 0.15), and the economic support coefficient is also non-significant ($\hat{\beta}_2$ = +0.213, *p* = 0.16). In the interaction specification, the stringency main effect strengthens ($\hat{\beta}_1$ = −1.654, *p* = 0.037), the economic support main effect remains null ($\hat{\beta}_2$ = −0.260, *p* = 0.40), and the interaction term is marginally non-significant ($\hat{\beta}_3$ = +1.328, *p* = 0.053). The marginal interaction suggests a slight tendency for economic support to attenuate the (negative) stringency effect, but the overall pattern does not support the economic buffer hypothesis. Critically, adding economic support controls does not reveal a hidden positive effect of stringency on polarization — the coefficient remains negative or null across all specifications. The null finding is not an artifact of omitted fiscal policy controls.

In a cross-sectional OLS regression of total polarization change (2019–2024) on mean stringency and mean economic support (2020–2021), stringency retains its significant negative association ($\hat{\beta}$ = −0.054, *p* = 0.008) while economic support is non-significant ($\hat{\beta}$ = +0.006, *p* = 0.36). Economic support measures — however important for individual welfare — did not detectably moderate the relationship between lockdowns and social polarization.

---

## SI Text 2: Controlling for GDP in the Selection Mechanism

A potential confounder of the selection effect (Figure 3, $r$ = −0.51) is national wealth. Richer countries tend to have both higher social capital and different governance capacities, raising the possibility that GDP per capita, not social cohesion, drives the association between pre-pandemic conditions and lockdown stringency. We test this hypothesis by estimating:

$$\text{Stringency}_i = \beta_1 \text{SocialCohesion}_i + \beta_2 \log(\text{GDP per capita})_i + \varepsilon_i$$

**Results.** In the bivariate specification, pre-pandemic social cohesion (v2smpolsoc, 2019) is a highly significant predictor of lockdown stringency ($\hat{\beta}_1$ = −2.429, *p* = 0.0003, $R^2$ = 0.260). When log GDP per capita is added as a control, the social cohesion coefficient is virtually unchanged ($\hat{\beta}_1$ = −2.500, *p* = 0.0006), while GDP per capita is substantively zero and non-significant ($\hat{\beta}_2$ = +0.542, *p* = 0.826). The $R^2$ does not improve (0.261 vs. 0.260). The partial correlation between social cohesion and stringency, net of GDP per capita, remains substantial ($r_{partial}$ = −0.465, *p* = 0.003).

Although social cohesion and GDP per capita are positively correlated in the OECD ($r$ = 0.510, *p* = 0.001), wealth does not independently predict lockdown strategy once social cohesion is accounted for. The selection effect operates through the social, not the economic, dimension of national development. Countries adopted stringent lockdowns because their populations lacked the mutual trust required for voluntary compliance, not because they were poor. This conclusion is consistent with the observation that several high-GDP OECD members (e.g., the United States, the United Kingdom) adopted strict lockdowns despite considerable national wealth, while lower-GDP high-trust nations (e.g., Estonia, Latvia) governed with lighter measures.

---

## SI Text 3: Sensitivity Analysis

We assess the robustness of the double null finding across alternative outcome measures, temporal specifications, and sample perturbations.

### Alternative Outcomes

The null stringency effect is not specific to the v2smpolsoc measure of social cohesion. Replacing the outcome with v2cacamps (political polarization of society), v2x_libdem (liberal democracy index), or v2x_jucon (judicial constraints on the executive) yields non-significant stringency coefficients in all cases (SI Figure S1). For political polarization, the stringency coefficient is −0.583 (*p* = 0.21). For liberal democracy, it is +0.012 (*p* = 0.69). For judicial constraints, it is +0.003 (*p* = 0.91). Mortality coefficients are similarly null across all outcomes. The pandemic policy response did not detectably affect any dimension of democratic governance or social polarization in the OECD.

### Temporal Sensitivity

If lockdown effects operate with a delay — for instance, if the polarizing consequences of coercive governance manifest only after mandates are lifted — the contemporaneous specification may miss them. We re-estimate the main model using lagged stringency at $t-1$ and $t-2$. The contemporaneous specification yields $\hat{\beta}$ = −0.785 (*p* = 0.22). The $t-1$ lag yields $\hat{\beta}$ = −0.110 (*p* = 0.83). The $t-2$ lag yields $\hat{\beta}$ = −0.183 (*p* = 0.69). All lagged specifications produce non-significant coefficients that remain negative or near zero (SI Figure S1), ruling out the possibility that our null result reflects a timing mismatch between lockdown exposure and polarization response.

### Jackknife Robustness (SI Table S1)

To confirm that the selection effect ($r$ = −0.51) is not driven by any single influential observation — such as Sweden (famously light-touch) or the United States (polarized and stringent) — we conduct a leave-one-out (jackknife) analysis, iteratively dropping each of the 38 OECD countries and recomputing the Pearson correlation between pre-pandemic social cohesion and lockdown stringency.

The correlation ranges from $r$ = −0.568 (dropping Hungary) to $r$ = −0.470 (dropping Chile), with a jackknife mean of −0.510 (SD = 0.021). All 38 iterations are significant at $p$ < 0.01. No single country drives the result. The most influential observations are Chile (whose removal attenuates the correlation most, to −0.470) and Hungary (whose removal strengthens it most, to −0.568). Both are consistent with their positions as off-diagonal cases — Chile is highly polarized with high stringency (on the regression line), while Hungary is polarized with low stringency (off the regression line, a populist outlier). Even in the most conservative iteration, the selection effect remains highly significant and substantively large.

---

## SI Table S1: Jackknife Robustness — Selection Effect

Leave-one-out cross-validation of the correlation between pre-pandemic social cohesion (v2smpolsoc, 2019) and mean lockdown stringency (2020–2021). Full sample: $r$ = −0.510, $p$ = 0.001, $N$ = 38.

| Dropped | Country | $r$ | $p$ |
|---------|---------|-----|-----|
| HUN | Hungary | −0.568 | 0.0002 |
| CAN | Canada | −0.566 | 0.0003 |
| POL | Poland | −0.538 | 0.0006 |
| EST | Estonia | −0.536 | 0.0006 |
| PRT | Portugal | −0.531 | 0.0007 |
| AUS | Australia | −0.529 | 0.0008 |
| IRL | Ireland | −0.523 | 0.0009 |
| KOR | South Korea | −0.522 | 0.0009 |
| CRI | Costa Rica | −0.519 | 0.0010 |
| CHE | Switzerland | −0.518 | 0.0010 |
| SVK | Slovakia | −0.517 | 0.0010 |
| DEU | Germany | −0.516 | 0.0011 |
| AUT | Austria | −0.515 | 0.0011 |
| SVN | Slovenia | −0.514 | 0.0011 |
| MEX | Mexico | −0.513 | 0.0012 |
| ITA | Italy | −0.512 | 0.0012 |
| NLD | Netherlands | −0.512 | 0.0012 |
| TUR | Turkiye | −0.511 | 0.0012 |
| BEL | Belgium | −0.510 | 0.0013 |
| FRA | France | −0.509 | 0.0013 |
| CZE | Czechia | −0.509 | 0.0013 |
| ISR | Israel | −0.507 | 0.0014 |
| LVA | Latvia | −0.505 | 0.0014 |
| GBR | United Kingdom | −0.505 | 0.0014 |
| SWE | Sweden | −0.504 | 0.0015 |
| ESP | Spain | −0.504 | 0.0015 |
| GRC | Greece | −0.503 | 0.0015 |
| USA | United States | −0.502 | 0.0016 |
| NOR | Norway | −0.501 | 0.0016 |
| LTU | Lithuania | −0.501 | 0.0016 |
| DNK | Denmark | −0.498 | 0.0017 |
| NZL | New Zealand | −0.490 | 0.0021 |
| LUX | Luxembourg | −0.489 | 0.0021 |
| JPN | Japan | −0.483 | 0.0025 |
| COL | Colombia | −0.482 | 0.0025 |
| FIN | Finland | −0.474 | 0.0031 |
| ISL | Iceland | −0.472 | 0.0032 |
| CHL | Chile | −0.470 | 0.0033 |

Range: [−0.568, −0.470]. All iterations significant at *p* < 0.01.

---

## SI Figure S1

**Figure S1.** Coefficient estimates for lockdown stringency across all model specifications. Points represent TWFE first-difference estimates with 95% confidence intervals (clustered standard errors at the country level). The vertical dashed line marks zero. Blue indicates the main specification (first-differenced social polarization as outcome); gray indicates robustness checks (alternative V-Dem outcomes, economic support controls, lagged stringency at $t-1$ and $t-2$). Across all seven specifications, the stringency coefficient is either statistically indistinguishable from zero or negative (indicating, if anything, reduced polarization). No specification yields a significant positive estimate, ruling out the hypothesis that lockdown stringency increased social or political polarization in OECD countries. File: `eipsa_si_forest_plot.png`

---

## SI Table S2: Full Regression Results with Economic Support Controls

| | Model A (Additive) | Model B (Interaction) |
|---|---|---|
| Stringency (0–1) | −0.913 (0.637) | −1.654** (0.789) |
| Economic Support (0–1) | +0.213 (0.153) | −0.260 (0.308) |
| Stringency $\times$ Econ. Support | — | +1.328* (0.683) |
| Entity FE | Yes | Yes |
| Year FE | Yes | Yes |
| Observations | 380 | 380 |
| $R^2$ (within) | −0.143 | −0.516 |

Standard errors (clustered by country) in parentheses. Dependent variable: $\Delta$ Social Polarization ($= -\Delta$v2smpolsoc). * $p$ < 0.10, ** $p$ < 0.05, *** $p$ < 0.01.

---

## SI Table S3: Selection Effect with GDP Control

| | Model 1 (Bivariate) | Model 2 (GDP Control) |
|---|---|---|
| Social Cohesion (v2smpolsoc) | −2.429*** (0.671) | −2.500*** (0.731) |
| log(GDP per capita) | — | +0.542 (2.468) |
| Constant | 51.77*** (0.72) | 45.78** (25.49) |
| $R^2$ | 0.260 | 0.261 |
| $N$ | 38 | 38 |

Robust standard errors (HC1) in parentheses. Dependent variable: Mean Lockdown Stringency (2020–2021). Partial correlation (cohesion–stringency | GDP): $r$ = −0.465, $p$ = 0.003. * $p$ < 0.10, ** $p$ < 0.05, *** $p$ < 0.01.
