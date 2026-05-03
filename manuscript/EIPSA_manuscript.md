# The Stress Test That Changed Nothing: Pre-pandemic Social Cohesion Determined Lockdown Strategy but Neither the Virus nor the Response Drove Polarization in OECD Nations

---

## Abstract

**Background.** The Parasite-Stress Theory of Values predicts that infectious disease threats drive societies toward authoritarianism and social division. The COVID-19 pandemic offered an unprecedented test of this hypothesis across developed democracies, yet it remains unclear whether the political disruptions observed during the pandemic were caused by the biological threat, the policy response, or neither — merely surfacing pre-existing structural conditions.

**Methods.** We conducted a panel data analysis of 38 OECD countries (2019–2024) integrating three datasets: the Varieties of Democracy Project (V-Dem v15) for social polarization indicators, the EM-DAT International Disaster Database for historical epidemic exposure, and the Oxford COVID-19 Government Response Tracker (OxCGRT) via Our World in Data for mortality and lockdown stringency measures. We employed two-way fixed-effects models with first-differenced outcomes and clustered standard errors, supplemented by cross-sectional analyses with robust inference.

**Results.** We report a double null finding. First, COVID-19 mortality did not predict changes in social polarization across OECD nations (*p* = 0.25). Second, lockdown stringency did not predict increased polarization either; the coefficient was negative and non-significant (*p* = 0.13), suggesting, if anything, regression to the mean rather than a polarizing effect of coercion. Social cohesion across the OECD has been declining steadily since at least 2010, and the pandemic did not detectably accelerate this trajectory. The primary finding is instead a selection effect: pre-pandemic social cohesion was strongly negatively correlated with subsequent lockdown stringency (*r* = −0.51, *p* = 0.001). Cohesive societies — notably Japan and the Nordic countries — achieved compliance voluntarily and adopted lighter restrictions. Polarized societies — including the United States, Italy, and Chile — lacked the social infrastructure for voluntary compliance and resorted to stringent mandates.

**Conclusion.** The pandemic served as a stress test that revealed, rather than created, the fractures in OECD democracies. Neither the virus nor the lockdowns drove social polarization; the decline in cohesion is structural and pre-dates COVID-19 by at least a decade. The critical policy variable — lockdown stringency — was itself a product of pre-existing social conditions, not an independent cause of political change. Pandemic preparedness in democracies is ultimately a function of social capital accumulated long before a crisis arrives.

---

## 1. Introduction

The relationship between infectious disease and political authoritarianism is among the most provocative hypotheses in evolutionary political psychology. The Parasite-Stress Theory of Values, advanced by Thornhill and Fincher (2014), posits that societies historically exposed to high pathogen prevalence develop stronger in-group conformity, reduced openness to outsiders, and greater deference to authority — behavioral adaptations that limit contagion at the cost of individual liberty. Cross-national evidence has been cited in support of this framework: regions with higher parasite stress tend to exhibit more authoritarian governance, greater ethnocentrism, and reduced interpersonal trust (Thornhill & Fincher, 2014; Murray & Schaller, 2010). However, the theory has faced sustained methodological challenge. Hackman and Hruschka (2013) demonstrated that the original statistical analyses do not withstand scrutiny when issues of non-independence and confounding are addressed. Pollet, Tybur, Frankenhuis, and Rickard (2014) warned that country-level correlations between pathogen prevalence and cultural values suffer from the ecological fallacy, with individual-level effects far weaker than aggregate patterns suggest. And Tybur et al. (2016), in a 30-nation study, found that pathogen avoidance relates specifically to sexual conservatism rather than to authoritarianism broadly — a far narrower effect than the theory predicts. Nevertheless, when COVID-19 emerged as a global pandemic in early 2020, the theoretical priors of the parasite-stress framework generated a clear prediction. Scholars warned of an impending wave of "pandemic authoritarianism," in which fear of contagion would accelerate democratic backsliding worldwide (Maerz, Lührmann, Lachapelle, & Edgell, 2020; Thomson & Ip, 2020). The ingredients appeared to be in place: emergency powers were invoked across democracies, civil liberties were curtailed, and executive discretion expanded with minimal legislative oversight (Edgell et al., 2021). Freedom House reported that democracy had declined for the fifteenth consecutive year, with pandemic-related restrictions compounding existing authoritarian pressures (Repucci & Slipowitz, 2021).

Yet the experience of advanced democracies complicates this narrative. Among OECD nations — where institutional safeguards are robust and democratic norms are deeply embedded — the pandemic did not produce the authoritarian consolidation that the parasite-stress framework might have predicted. Democratic backsliding in the developed world was already a subject of scholarly concern before COVID-19, driven by populist mobilization, affective polarization, and declining trust in institutions (Norris & Inglehart, 2019; Levitsky & Ziblatt, 2018). Affective polarization — the tendency to view political opponents with hostility independent of policy disagreement — had been rising across Western democracies since the 1980s (Iyengar, Lelkes, Levendusky, Malhotra, & Westwood, 2019), and Boxell, Gentzkow, and Shapiro (2022) documented its cross-country spread across twelve OECD nations over decades, establishing it as a structural phenomenon long predating the pandemic. The pandemic intersected with these pre-existing trends, but whether it independently accelerated them remains an open empirical question. More precisely: was it the *biological threat* — the fear of illness and death — that eroded democratic norms? Or was it the *policy response* — the imposition of lockdowns, curfews, and mandates — that deepened social divisions?

This distinction is consequential but rarely operationalized. Much of the existing literature on COVID-19 and democracy treats the virus and the response as a single exogenous shock, examining composite indices of "pandemic impact" without disentangling mortality from mitigation (Lührmann & Lindberg, 2019; Maerz et al., 2020). Studies that do differentiate between the two tend to focus on individual-level survey data — attitudes toward democracy, trust in government, compliance behavior (Van Bavel et al., 2020) — rather than on country-level institutional and social outcomes measured longitudinally. A rigorous panel analysis that separately estimates the effects of pathogen exposure (mortality) and policy coercion (lockdown stringency) on democratic and social outcomes across high-income democracies remains, to our knowledge, absent from the literature.

This paper addresses that gap. Drawing on panel data from 38 OECD countries over the period 2019–2024, we integrate three datasets: the Varieties of Democracy Project (V-Dem v15; Coppedge et al., 2025) for democratic governance and social polarization indicators; the EM-DAT International Disaster Database (Guha-Sapir et al., 2025) for historical epidemic exposure; and the Oxford COVID-19 Government Response Tracker (OxCGRT; Hale et al., 2021) for mortality and lockdown stringency measures obtained via Our World in Data (Mathieu et al., 2021). Using two-way fixed-effects models with first-differenced outcomes and clustered standard errors, we separately estimate the effects of COVID-19 mortality and lockdown stringency on changes in social polarization.

Our findings reveal a double null result and a selection effect. First, COVID-19 mortality — the proximate operationalization of pathogen-driven fear — does not predict changes in social polarization across OECD nations. The parasite-stress mechanism, whatever its validity over evolutionary timescales, does not appear to operate through pandemic mortality in contemporary developed democracies. Second, lockdown stringency does not predict increased polarization either; the coefficient is negative and non-significant, inconsistent with the claim that coercive pandemic governance deepened social divisions. Third, and most critically, we demonstrate that pre-pandemic social cohesion — measured by V-Dem's social polarization index in 2019 — was strongly negatively correlated with subsequent lockdown stringency (*r* = −0.51, *p* = 0.001). Societies with high pre-existing cohesion, notably the Nordic countries and Japan, achieved pandemic compliance through voluntary behavioral adaptation, requiring minimal legal coercion. Societies with low cohesion, including the United States, Italy, and Chile, were compelled to substitute trust with mandates. Lockdown stringency was not a cause of polarization; it was a consequence of it.

The remainder of this paper proceeds as follows. Section 2 reviews the theoretical and empirical literature linking pathogen stress to political outcomes. Section 3 describes the data and empirical strategy. Section 4 presents the main results, including the historical epidemic null, the COVID-19 mortality null, the stringency null, and the selection mechanism. Section 5 discusses the implications for democratic governance in the post-pandemic era, and Section 6 concludes.

---

## 3. Data and Methods

### 3.1 Data Sources

This study integrates three publicly available datasets to construct a country-year panel of 38 OECD member states over the period 2019–2024.

**Outcome variables.** Democratic governance and social polarization indicators are drawn from the Varieties of Democracy Project, version 15 (V-Dem v15; Coppedge et al., 2025). Our primary outcome is *social polarization* (`v2smpolsoc`), which captures the degree to which society is divided along major political and social cleavages, with higher values indicating greater cohesion and lower values indicating deeper polarization. We additionally tested *liberal democracy* (`v2x_libdem`) and *political polarization* (`v2cacamps`) as alternative outcomes; both yielded null results and are reported for completeness.

**Exposure variables.** We operationalize two distinct dimensions of pandemic impact. The *biological threat* is measured as cumulative COVID-19 deaths per million population, log-transformed as ln(1 + deaths per million), sourced from Our World in Data (OWID; Mathieu et al., 2021), which aggregates data from the Johns Hopkins CSSE and national health agencies. The *policy response* is measured using the Stringency Index (0–100) from the Oxford COVID-19 Government Response Tracker (OxCGRT; Hale et al., 2021), a composite of nine containment and closure indicators including school closures, workplace closures, stay-at-home requirements, and restrictions on gatherings. We compute the country-level mean of the Stringency Index over 2020–2021, the period of maximum policy variation across OECD nations.

**Historical epidemic exposure.** To situate the COVID-19 analysis within a broader epidemiological context, we draw on the EM-DAT International Disaster Database (Guha-Sapir et al., 2025), which records 888 epidemic events across 604 country-years in the period 2000–2025. This dataset is used in preliminary models to test whether historical epidemic exposure — independent of COVID-19 — predicts political outcomes.

**Pre-pandemic baseline.** Pre-pandemic social cohesion is measured as the 2019 value of `v2smpolsoc`, providing a pre-treatment benchmark uncontaminated by pandemic dynamics. This variable serves both as a control for baseline conditions and as the key predictor in our selection analysis.

### 3.2 Sample

The sample comprises all 38 current OECD member states, providing a relatively homogeneous set of high-income democracies that nonetheless exhibit substantial variation in both pandemic policy responses (Stringency Index range: 37.5–65.3) and social cohesion (v2smpolsoc range: −2.87 to +2.48 in 2019). All countries have complete V-Dem coverage through 2024 and OWID coverage through at least mid-2024.

### 3.3 Empirical Strategy

Our identification strategy rests on a Two-Way Fixed Effects (TWFE) model estimated on first-differenced outcomes. For country *i* in year *t*, the estimating equation is:

$$\Delta Y_{it} = \beta X_{it} + \alpha_i + \gamma_t + \varepsilon_{it}$$

where $\Delta Y_{it} = Y_{it} - Y_{it-1}$ is the annual change in social polarization, $X_{it}$ is the exposure variable (log COVID-19 mortality or lockdown stringency), $\alpha_i$ denotes country fixed effects absorbing time-invariant confounders (political culture, institutional heritage, geography), and $\gamma_t$ denotes year fixed effects absorbing common temporal shocks (global pandemic waves, economic cycles, information environment shifts). Standard errors are clustered at the country level to account for serial correlation within panels.

The first-differencing of the outcome serves a specific purpose: it shifts the estimand from *levels* of polarization — which are dominated by slow-moving, path-dependent processes — to *changes* in polarization, isolating year-on-year acceleration that might plausibly respond to pandemic-era variation. This is a conservative specification; it asks not whether more-exposed countries *are* more polarized, but whether they *became more polarized faster* during the period of exposure.

We supplement the panel analysis with cross-sectional OLS regressions estimated on the 38-country sample, regressing the total change in social polarization (2019 to 2024) on mean lockdown stringency (2020–2021). These cross-sectional models sacrifice the temporal variation exploited by the panel but offer transparency and direct interpretability. Robust standard errors (HC1) are employed throughout.

To address the threat of reverse causality — the possibility that pre-existing polarization determined lockdown stringency rather than the reverse — we conduct a partial regression analysis, residualizing both the exposure and the outcome on 2019 baseline polarization. This Frisch-Waugh-Lovell decomposition isolates the component of the stringency–polarization association that is orthogonal to pre-pandemic conditions.

---

## 4. Results

### 4.0 Preliminary Analysis: Historical Epidemic Exposure

Before turning to the COVID-19 pandemic, we examined whether historical epidemic exposure in the pre-COVID era predicted social cohesion in OECD nations. Using the EM-DAT International Disaster Database (Guha-Sapir et al., 2025), we identified 23 epidemic events affecting 16 of 38 OECD countries between 2000 and 2019. The frequency of historical epidemics was not significantly correlated with either the level of social cohesion in 2019 (*r* = −0.30, *p* = 0.069) or the subsequent change in cohesion from 2019 to 2024 (*r* = +0.08, *p* = 0.627). Countries that had experienced epidemics did not differ significantly in baseline cohesion from those that had not (*t* = −1.67, *p* = 0.103). These null results reinforce the central finding of this paper: pathogen exposure — whether historical or contemporary — does not appear to be a primary driver of social polarization in established OECD democracies. This is consistent with Pasin et al. (2024), who found across 43 countries that macro-cultural dimensions remained unchanged in the short-term aftermath of COVID-19 despite theoretical predictions to the contrary. The forces shaping social cohesion in these nations operate through structural and institutional channels that are largely orthogonal to epidemiological history.

### 4.1 The Null Effect of Biological Threat

We begin with the prediction derived from Parasite-Stress Theory: that COVID-19 mortality — the direct biological threat — should predict democratic erosion or increased polarization. Table 1 (Model 1) reports the TWFE first-difference estimate of log COVID-19 mortality on changes in social polarization across all 38 OECD countries. The estimated coefficient is substantively small and statistically indistinguishable from zero ($\hat{\beta}$ = +0.023, *p* = 0.41). In the horse-race specification that includes both mortality and stringency (Model 3), the mortality estimate remains null ($\hat{\beta}$ = +0.032, *p* = 0.25). No specification — including models with lagged exposure — yields a significant association at conventional thresholds.

We interpret this result as inconsistent with the proximate application of Parasite-Stress Theory to contemporary OECD democracies. Whatever evolutionary mechanisms link pathogen exposure to authoritarian preferences, they do not appear to operate through pandemic mortality in societies with robust healthcare systems and established democratic institutions.

### 4.2 The Null Effect of Policy Stringency

If the virus itself did not drive political change, did the policy response? Table 1 (Models 2 and 3) reports estimates substituting lockdown stringency for COVID-19 mortality as the exposure variable.

The results do not support the hypothesis that lockdowns caused polarization. In the TWFE first-difference panel, the stringency coefficient is *negative* — the opposite of the predicted direction — and statistically non-significant (Model 2: $\hat{\beta}$ = −0.785, *p* = 0.22; Model 3: $\hat{\beta}$ = −0.944, *p* = 0.13). If taken at face value, the negative sign would imply that years with higher stringency saw *smaller* increases in polarization, not larger ones. However, given the wide confidence intervals — which comfortably span zero in both specifications — the most parsimonious interpretation is that lockdown stringency had no detectable effect on the trajectory of social polarization within OECD countries.

This null result is noteworthy in light of the prominent public discourse linking lockdowns to social division. While lockdowns may have generated salient political conflicts in individual countries — mask mandates in the United States, vaccine pass protests in France, freedom convoys in Canada — these episodic disruptions do not appear to have moved the aggregate polarization index at the country-year level. The within-country variation in stringency over time does not predict within-country changes in social cohesion.

Figure 1 provides important context for this null finding. Social cohesion across the OECD has been declining steadily since at least 2010, falling from a mean v2smpolsoc of +0.52 in 2010 to −0.26 in 2024 — a cumulative decline of 0.79 units over fifteen years. The pandemic period (2020–2024) shows no visible inflection in this trend. The decline from 2019 (−0.08) to 2024 (−0.26) is a continuation of the same slope observed in the pre-pandemic decade, not an acceleration. The pandemic, for all its disruption, did not bend the curve of polarization in OECD democracies.

In the cross-sectional residualized specification (Model 4), which controls for 2019 baseline cohesion via the Frisch-Waugh-Lovell decomposition, the stringency coefficient is again negative and only marginally significant ($\hat{\beta}$ = −0.031, *p* = 0.10). This marginal result is more plausibly interpreted as residual mean reversion — countries that were more polarized in 2019 (which also adopted stricter lockdowns) regressed partially toward the mean — than as evidence that lockdowns improved social cohesion.

### 4.3 The Selection Effect: Social Cohesion Determined the Policy Response

Having established that neither the virus nor the lockdowns drove polarization, we turn to the question of what determined the policy response itself. Figure 3 plots pre-pandemic social cohesion (V-Dem v2smpolsoc, 2019) against mean lockdown stringency (2020–2021) for all 38 OECD countries. The correlation is strongly negative ($r$ = −0.510, *p* = 0.001): countries that entered the pandemic with greater social cohesion subsequently adopted *less* stringent lockdown policies.

This finding reframes the entire analytical question. The relationship between lockdown stringency and political outcomes is not a story of policy causing polarization; it is a story of polarization shaping policy. Pre-pandemic social conditions selected countries into distinct governance regimes, and this selection explains the cross-sectional association between stringency and polarization that a naive analysis might interpret as causal.

The selection partitions the OECD into two governance archetypes. In the "voluntary compliance" quadrant — high cohesion, low stringency — we find Japan, Finland, Denmark, Iceland, New Zealand, Norway, and Sweden. These societies possessed the social capital to coordinate behavioral change without recourse to legal mandates. Citizens voluntarily adopted protective behaviors — masking, distancing, self-isolation — not because of government orders but because of shared norms of collective responsibility. Stringency indices remained low not because these governments were indifferent to public health, but because formal coercion was unnecessary.

In the "coercive response" quadrant — low cohesion, high stringency — we find Chile, Italy, Greece, Colombia, the United Kingdom, the United States, Spain, and Turkey. In these societies, the absence of sufficient social trust meant that voluntary compliance could not be assumed. Governments faced a constrained choice set: either tolerate non-compliance and its epidemiological consequences, or impose mandates backed by legal authority. Most chose the latter. The stringency index in these countries reflects not greater governmental ambition but the structural necessity of substituting formal rules for absent informal norms.

Off-diagonal cases enrich the pattern. Canada, Australia, and Ireland (high cohesion, high stringency) adopted strict measures despite adequate social capital — plausibly driven by geographic isolation, early-wave severity, or precautionary governance cultures. Hungary and Poland (low cohesion, low stringency) avoided strict lockdowns despite deep polarization, consistent with populist governments that instrumentalized pandemic skepticism for political gain. South Korea (low cohesion, moderate stringency) achieved containment through technological surveillance rather than either trust or coercion, representing a distinct governance pathway.

The selection effect also explains the marginal negative coefficient observed in the residualized specification (Model 4, *p* = 0.10). After partialling out 2019 baseline cohesion, the residual association between stringency and polarization change is slightly negative — meaning that countries with unexpectedly high stringency (given their baseline) saw slightly *less* polarization growth. This pattern is consistent with mean reversion rather than a causal protective effect of lockdowns: countries far from the mean in 2019 tended to move back toward it by 2024, regardless of their lockdown strategy.

---

## 5. Discussion

### 5.1 Parasite-Stress Theory Revisited

The Parasite-Stress Theory of Values predicts that infectious disease threats drive societies toward authoritarianism, in-group conformity, and reduced tolerance for dissent (Thornhill & Fincher, 2014). Our findings challenge the applicability of this framework to contemporary high-income democracies on two fronts. First, COVID-19 mortality bore no detectable relationship to changes in social polarization — the biological threat channel is null. This is consistent with Pasin et al. (2024), who examined survey data from 29,761 individuals across 43 countries before and after the onset of COVID-19, finding that macro-cultural dimensions — collectivism, conformity, outgroup prejudice — remained unchanged despite the most severe global pathogen shock in a century. Second, the coercive policy instruments deployed in response to the pandemic — instruments that might plausibly function as a proxy for the "authoritarian shift" the theory predicts — also failed to move the needle. As Hackman and Hruschka (2013) and Pollet et al. (2014) have argued on methodological grounds, the parasite-stress framework may describe long-run ecological correlations without identifying a causal mechanism that operates in real time. Our results provide a direct empirical confirmation of this critique.

This double null result does not invalidate the evolutionary logic of the theory. It does, however, establish a clear boundary condition: in societies with robust democratic institutions, independent judiciaries, free media, and high baseline life expectancy, neither pathogen fear nor the emergency powers it occasions appear sufficient to reshape the social fabric. Modern institutions absorb both the psychological shock of disease and the political shock of emergency governance without transmitting either into lasting structural change.

### 5.2 The Pandemic as Stress Test

If the pandemic did not cause polarization, what did it do? Our findings suggest it functioned as a *diagnostic* — a stress test that revealed the load-bearing capacity of each democracy's social infrastructure, without itself adding to the structural load.

The selection effect documented in Figure 3 ($r$ = −0.51, *p* = 0.001) is the central evidence for this interpretation. The single strongest predictor of a country's lockdown strategy was not its epidemiological situation, its government's ideology, or its healthcare capacity — it was the level of social trust its citizens held toward one another before the crisis began. This finding reframes lockdown stringency from a policy *choice* to a structural *revelation*. Countries did not choose high stringency; their social conditions chose it for them.

For high-trust nations — Japan, Finland, Denmark, New Zealand, and Sweden — low lockdown stringency should not be interpreted as policy laxity or governmental indifference. Rather, it reflected what Fukuyama (1995) termed the capacity of high-trust societies to achieve collective outcomes at lower transaction costs — a principle he applied directly to the pandemic context, arguing that social trust, not regime type, was the critical variable separating successful from unsuccessful responses (Fukuyama, 2020). This mechanism is now empirically well-documented. Bargain and Aminjonov (2020) show that European regions with higher institutional trust exhibited significantly greater voluntary reductions in mobility during the pandemic, independent of the stringency of formal mandates. Durante, Guiso, and Gulino (2021) find that Italian municipalities with stronger civic traditions practiced social distancing *before* government orders were issued — trust preceded coercion rather than following it. At the aggregate level, Bartscher, Seitz, Siegloch, Slotwinski, and Wehrhöfer (2021) demonstrate that social capital predicted slower COVID-19 transmission across European countries through precisely this voluntary compliance channel. These findings operationalize the theoretical insight of Van Bavel et al. (2020), who argued in their landmark review that social norms and trust in institutions are among the most powerful determinants of health behavior during pandemics. The nations in our "voluntary compliance" quadrant possessed these resources in abundance. Their light-touch governance was not a failure of state capacity but its highest expression: the ability to coordinate a population-level behavioral response through shared norms and mutual obligation, without recourse to the blunt instrument of legal enforcement.

For low-trust nations — the United States, Italy, Chile, Turkey — the pandemic revealed a deficit that predated the crisis by years. These societies could not generate voluntary compliance because the prerequisite — mutual trust across social and political lines — was already depleted. Coercion was not a policy failure; it was a structural necessity. The mandates, curfews, and enforcement actions that characterized these countries' pandemic responses were symptoms of low social capital, not causes of its further erosion. Crucially, these countries were not polarized *because* they locked down hard. They locked down hard *because* they were polarized.

This distinction has implications for how we conceptualize state capacity. The conventional understanding of state capacity emphasizes the ability to enforce compliance — to tax, to regulate, to coerce. Our findings suggest a complementary dimension that might be termed *relational capacity*: the ability to elicit voluntary cooperation in the absence of formal enforcement. By this measure, the most capable states during the pandemic were not those with the strongest enforcement apparatus but those that never needed to deploy it.

### 5.3 The Stubbornness of Structural Polarization

Perhaps the most striking finding is not what the pandemic caused but what it failed to alter. Figure 1 documents a steady, monotonic decline in social cohesion across the OECD from 2010 to 2024 — a decline that the pandemic neither accelerated nor arrested. The mean v2smpolsoc dropped by 0.79 units over fifteen years, at a roughly constant rate, with no visible inflection at 2020.

This finding speaks to the structural nature of polarization in advanced democracies. The forces driving social division — economic inequality, cultural sorting, media fragmentation, algorithmic amplification, declining institutional trust — operate on timescales and through mechanisms that a single exogenous shock, however severe, cannot easily override. Iyengar et al. (2019) trace the rise of affective polarization in the United States to partisan sorting processes unfolding over four decades, while McCarty, Poole, and Rosenthal (2016) demonstrate that economic inequality and elite ideological divergence are its primary structural drivers. Boxell, Gentzkow, and Shapiro (2017) add the provocative finding that polarization in the United States grew fastest among demographic groups *least* likely to use the internet, challenging the intuition that social media is the proximate cause and reinforcing the primacy of deeper structural forces. If these slow-moving fundamentals are the true engines of polarization, it follows that a pandemic — however socially disruptive in the short term — would leave the trajectory largely undisturbed. This is precisely what Figure 1 documents.

### 5.4 Limitations

Several limitations warrant acknowledgment. First, our sample is restricted to 38 OECD member states. While this provides a coherent set of high-income democracies, it limits generalizability to developing nations and autocracies, where both the biological impact and the political dynamics of the pandemic may differ substantially. Second, the analysis remains observational. The selection effect is compelling but correlational; we cannot exclude the possibility that unobserved variables drive both pre-pandemic cohesion and lockdown strategy simultaneously. Third, V-Dem indicators are expert-coded annual assessments that may not capture rapid, within-year fluctuations in social sentiment. Survey-based measures of interpersonal trust and political attitudes might reveal dynamics invisible at the annual institutional level. Fourth, the Stringency Index is a composite measure that aggregates diverse policy instruments — school closures, workplace restrictions, travel bans — which may have heterogeneous effects. Finally, the negative within-R-squared values in some TWFE specifications indicate that the fixed effects absorb most meaningful variation, leaving limited identifying variation for the exposure variables — a common challenge in short panels with slow-moving outcomes.

---

## 6. Conclusion

This paper set out to test whether the COVID-19 pandemic drove social polarization in the developed world, and if so, through what mechanism — the virus or the lockdowns. The answer is neither.

Across 38 OECD countries over the period 2019–2024, we find no evidence that COVID-19 mortality predicted changes in social cohesion. Nor do we find evidence that lockdown stringency accelerated polarization; the coefficient is negative and non-significant, inconsistent with the claim that coercive pandemic governance deepened social divisions. The decline in social cohesion observed across the OECD is structural, pre-dating the pandemic by at least a decade, and continuing through it at an unaltered pace. The pandemic, for all its disruption, did not detectably change the political trajectory of these societies.

What the pandemic did reveal was the asymmetry of democratic governance under crisis. The single strongest predictor of lockdown stringency was not the severity of the outbreak but the level of pre-existing social cohesion (*r* = −0.51, *p* = 0.001). Societies rich in social capital — Japan, the Nordic countries, New Zealand — could afford to govern lightly, achieving public health objectives through voluntary compliance. Societies poor in social capital — the United States, Southern Europe, Latin America — were compelled to govern heavily, not because their leaders chose authoritarianism but because their social conditions left no alternative. Coercion was not the cause of distrust; it was its consequence.

The implication for future crisis preparedness is sobering. Pandemic planning has overwhelmingly focused on medical infrastructure: surveillance systems, vaccine pipelines, hospital surge capacity, stockpiles of personal protective equipment. Our findings suggest that the binding constraint on effective democratic crisis response is not medical but social. The countries that navigated the pandemic with the least friction were not those with the best epidemiological preparedness but those with the deepest reserves of mutual trust. Social capital, unlike ventilators, cannot be manufactured in an emergency. It is accumulated slowly, through decades of institutional performance, equitable governance, and civic participation — and once depleted, it is not easily restored. If the next global crisis arrives in a society already fractured by distrust, the same constrained choice will recur: tolerate the consequences of non-compliance, or impose mandates that confirm and deepen the very alienation they seek to overcome. The pandemic changed nothing — and that is precisely the problem.

---

## References

Amat, F., Arenas, A., Falcó-Gimeno, A., & Muñoz, J. (2020). Pandemics meet democracy: Experimental evidence from the COVID-19 crisis in Spain. *SocArXiv Preprint*. https://doi.org/10.31235/osf.io/dkusw

Bargain, O., & Aminjonov, U. (2020). Trust and compliance to public health policies in times of COVID-19. *Journal of Public Economics*, *192*, 104316. https://doi.org/10.1016/j.jpubeco.2020.104316

Bartscher, A. K., Seitz, S., Siegloch, S., Slotwinski, M., & Wehrhöfer, N. (2021). Social capital and the spread of Covid-19: Insights from European countries. *Journal of Health Economics*, *80*, 102531. https://doi.org/10.1016/j.jhealeco.2021.102531

Boxell, L., Gentzkow, M., & Shapiro, J. M. (2017). Greater internet use is not associated with faster growth in political polarization among US demographic groups. *Proceedings of the National Academy of Sciences*, *114*(40), 10612–10617. https://doi.org/10.1073/pnas.1706588114

Boxell, L., Gentzkow, M., & Shapiro, J. M. (2022). Cross-country trends in affective polarization. *Review of Economics and Statistics*, *106*(2), 557–565. https://doi.org/10.1162/rest_a_01160

Cashdan, E., & Steele, M. (2013). Pathogen prevalence, group bias, and collectivism in the standard cross-cultural sample. *Human Nature*, *24*(1), 59–75. https://doi.org/10.1007/s12110-012-9159-3

Coppedge, M., Gerring, J., Knutsen, C. H., Lindberg, S. I., Teorell, J., Altman, D., Bernhard, M., Cornell, A., Fish, M. S., Gastaldi, L., Gjerløw, H., Glynn, A., God, A. G., Grahn, S., Hicken, A., Kinzelbach, K., Marquardt, K. L., McMann, K., Mechkova, V., ... Ziblatt, D. (2025). *V-Dem dataset v15* [Data set]. Varieties of Democracy (V-Dem) Project. https://doi.org/10.23696/vdemds25

Durante, R., Guiso, L., & Gulino, G. (2021). Asocial capital: Civic culture and social distancing during COVID-19. *Journal of Public Economics*, *194*, 104342. https://doi.org/10.1016/j.jpubeco.2020.104342

Edgell, A. B., Grahn, S., Lachapelle, J., Lührmann, A., & Maerz, S. F. (2021). An update on pandemic backsliding: Democracy four months after the beginning of the Covid-19 pandemic. *V-Dem Policy Brief*, 24. V-Dem Institute, University of Gothenburg.

Fukuyama, F. (1995). *Trust: The social virtues and the creation of prosperity*. Free Press.

Fukuyama, F. (2020). The pandemic and political order. *Foreign Affairs*, *99*(4), 26–32.

Guha-Sapir, D., Below, R., & Hoyois, Ph. (2025). *EM-DAT: The international disaster database* [Data set]. Centre for Research on the Epidemiology of Disasters (CRED), UCLouvain. https://www.emdat.be

Hackman, J., & Hruschka, D. (2013). Analyses do not support the parasite-stress theory of human sociality. *Behavioral and Brain Sciences*, *36*(2), 83–85. https://doi.org/10.1017/S0140525X12000828

Hale, T., Angrist, N., Goldszmidt, R., Kira, B., Petherick, A., Phillips, T., Webster, S., Cameron-Blake, E., Hallas, L., Majumdar, S., & Tatlow, H. (2021). A global panel database of pandemic policies (Oxford COVID-19 Government Response Tracker). *Nature Human Behaviour*, *5*(4), 529–538. https://doi.org/10.1038/s41562-021-01079-8

Iyengar, S., Lelkes, Y., Levendusky, M., Malhotra, N., & Westwood, S. J. (2019). The origins and consequences of affective polarization in the United States. *Annual Review of Political Science*, *22*, 129–146. https://doi.org/10.1146/annurev-polisci-051117-073034

Levitsky, S., & Ziblatt, D. (2018). *How democracies die*. Crown.

Lührmann, A., & Lindberg, S. I. (2019). A third wave of autocratization is here: What is new about it? *Democratization*, *26*(7), 1095–1113. https://doi.org/10.1080/13510347.2019.1582029

Maerz, S. F., Lührmann, A., Lachapelle, J., & Edgell, A. B. (2020). Worth the sacrifice? Illiberal and authoritarian practices during Covid-19. *V-Dem Working Paper*, 110. V-Dem Institute.

Mathieu, E., Ritchie, H., Rodés-Guirao, L., Appel, C., Giattino, C., Hasell, J., Macdonald, B., Dattani, S., Beltekian, D., Ortiz-Ospina, E., & Roser, M. (2021). A global database of COVID-19 vaccinations. *Nature Human Behaviour*, *5*(7), 947–953. https://doi.org/10.1038/s41562-021-01122-8

McCarty, N., Poole, K. T., & Rosenthal, H. (2016). *Polarized America: The dance of ideology and unequal riches* (2nd ed.). MIT Press.

Murray, D. R., & Schaller, M. (2010). Historical prevalence of infectious diseases within 230 geopolitical regions: A tool for investigating origins of culture. *Journal of Cross-Cultural Psychology*, *41*(1), 99–108. https://doi.org/10.1177/0022022109349510

Norris, P., & Inglehart, R. (2019). *Cultural backlash: Trump, Brexit, and authoritarian populism*. Cambridge University Press. https://doi.org/10.1017/9781108595841

Pasin, G. L., Szekely, A., Eriksson, K., Gelfand, M., Biber, P., Romano, A., ... & Spadaro, G. (2024). Evidence from 43 countries that disease leaves cultures unchanged in the short-term. *Scientific Reports*, *14*, 6502. https://doi.org/10.1038/s41598-023-33155-6

Pollet, T. V., Tybur, J. M., Frankenhuis, W. E., & Rickard, I. J. (2014). What can cross-cultural correlations teach us about human nature? *Human Nature*, *25*(3), 410–429. https://doi.org/10.1007/s12110-014-9206-3

Putnam, R. D. (2000). *Bowling alone: The collapse and revival of American community*. Simon & Schuster.

Repucci, S., & Slipowitz, A. (2021). *Freedom in the World 2021: Democracy under siege*. Freedom House.

Thomson, S., & Ip, E. C. (2020). COVID-19 emergency measures and the impending authoritarian pandemic. *Journal of Law and the Biosciences*, *7*(1), lsaa064. https://doi.org/10.1093/jlb/lsaa064

Thornhill, R., & Fincher, C. L. (2014). *The parasite-stress theory of values and sociality: Infectious disease, history and human values worldwide*. Springer. https://doi.org/10.1007/978-3-319-08040-6

Tybur, J. M., Inbar, Y., Aarøe, L., Barclay, P., Barlow, F. K., de Barra, M., ... & Žeželj, I. (2016). Parasite stress and pathogen avoidance relate to distinct dimensions of political ideology across 30 nations. *Proceedings of the National Academy of Sciences*, *113*(44), 12408–12413. https://doi.org/10.1073/pnas.1607398113

Van Bavel, J. J., Baicker, K., Boggio, P. S., Capraro, V., Cichocka, A., Cikara, M., ... & Willer, R. (2020). Using social and behavioural science to support COVID-19 pandemic response. *Nature Human Behaviour*, *4*(5), 460–471. https://doi.org/10.1038/s41562-020-0884-z

---

## Figures and Tables

**Figure 1.** OECD Social Cohesion Trend, 2010–2024. Mean V-Dem v2smpolsoc across 38 OECD countries with 95% confidence band. The steady decline from +0.52 (2010) to −0.26 (2024) shows no inflection at the pandemic onset (2020). File: `eipsa_figure1.png`

**Figure 2.** Coefficient Plot: Horse-Race Model (Model 3). Point estimates and 95% confidence intervals for COVID-19 mortality and lockdown stringency predicting changes in social polarization. Both coefficients span zero. File: `eipsa_figure2.png`

**Figure 3.** The Selection Effect: Pre-pandemic Social Cohesion vs. Lockdown Stringency across 38 OECD Countries. Scatter plot with OLS trend line and 95% confidence interval. Dashed lines indicate sample medians, defining four governance quadrants. *r* = −0.510, *p* = 0.001. File: `eipsa_figure3.png`

**Table 1.** Two-Way Fixed-Effects Regression Results: Pandemic Exposure and Changes in Social Polarization. Four models: (1) Biological Threat, (2) Policy Response, (3) Horse Race, (4) Selection (Residualized). See `eipsa_final_manuscript.py` for formatted output.
