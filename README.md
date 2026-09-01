# CrashGap

Female-vs-male fatality risk in the *same* crash, computed live from public-domain NHTSA
[FARS](https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars) data — with
the seat-position confound separated out instead of quietly baked in.

Cars are crash-tested around the median male body. The literature has measured the consequence for
decades; the number just isn't anywhere you can link to, query, or watch move as new model years
arrive. CrashGap is that missing layer: an annual pipeline, a SQLite `results` table where every
estimate carries its run timestamp, git commit, FARS years and full serialized cohort definition,
and a dashboard that leads with the decomposition rather than the headline-friendly number.

## The headline, and why it isn't the obvious one

Restrict to belted front-seat **mixed-sex pairs in the same vehicle** in a fatal frontal crash and
pool them, and women come out **1.09×** more likely to be the one who died (FARS 2015–2024, 22,721
pairs). That number is wrong to publish, and here is why: seat position is the one attribute that
differs within every pair, and it is ~73/27 collinear with sex — she is the passenger in 16,656 of
those pairs and the driver in only 6,065. Split by seat configuration and the ratio *flips*:

| configuration | pairs | she died : he died |
|---|---:|---:|
| she is the passenger, he drives | 16,656 | 1.153 |
| she drives, he is the passenger | 6,065 | 0.940 |

If sex alone drove the outcome those would agree. Evans' geometric means separate them — with *f*
the female multiplier and *s* the right-front multiplier, the two configurations measure *f·s* and
*f/s*, so `sqrt(product)` recovers *f* and `sqrt(quotient)` recovers *s*:

| effect | frontal core | 95% CI | significant |
|---|---:|---|---|
| being female (seat-balanced) | **+4.1%** | −1.5% to +10.1% | no |
| sitting right-front | **+10.8%** | +4.7% to +17.1% | yes |

So on FARS 2015–2024, in the primary frontal cut, **the seat matters more than the sex**, and the
1.09 pooled figure is mostly a seat effect wearing a sex label.

**The cuts disagree, and that's informative.** In the strict head-on cut (`MAN_COLL=2`) the female
effect is **+11.5% and significant**, while the seat effect *reverses* — the driver fares worse.
That's physically plausible rather than noise: head-on collisions are predominantly offset toward
the driver's side, which is why the driver-side small-overlap crash test existed for six years
before a passenger-side one was added. The honest summary is that the seat-balanced female effect
sits somewhere from low single digits to low double digits depending on crash geometry, and ten
years of FARS can't pin it down further. Quoting only the significant row is cherry-picking.

None of this refutes the larger published severity-adjusted gap (Bose 2011 reports MAIS 3+ odds
~1.47). That's a different quantity, measured on crash-investigation data with injury grading and
survey weights — the CISS rung, not this one.

## Method

Evans double-pair design: only vehicles carrying exactly one eligible male and one eligible female
(both belted, front outboard, age 16–96, light vehicle) enter, which holds crash severity, delta-V,
vehicle and impact direction physically constant within the vehicle. FARS is a census of *fatal*
crashes, so a raw occupant-level odds ratio would be dominated by crash-involvement selection and
can point the wrong way — hence the within-vehicle design, and hence the seat split on top of it.
Frontal is pinned to IMPACT1 ∈ {11, 12, 1} (Forman 2019), with the wider fan and head-on cut
reported as sensitivity variants. Every code set is a literal in `crashgap/codebook.py` with its
source beside it. Confidence intervals are percentile bootstraps resampling **whole crashes**,
never persons, with both seat strata resampled inside one replicate.

What this repo adds over the literature (Evans 1988; Bose 2011; Forman 2019; Atwood, Noh & Craig
2023) is the live, versioned, machine-readable form — not new evidence, and it says so on the page.

What these numbers are **not**: population rates (every crash here already killed someone),
severity-adjusted odds ratios, age-adjusted (women in these pairs average ~1.4 years younger), or
evidence about the model-year trend (this window can't resolve it).

## v1: age-adjusted regression and the seat × sex interaction

v0's double-pair decomposition can't separate age from sex, and can't say anything about the
model-year trend or whether the seat effect itself differs by sex — those need a real regression,
not a ratio of geometric means. v1 adds `statsmodels` (a new runtime dependency; `Logit` is the only
call site) and fits a **within-vehicle differenced conditional logit**: for a 2-occupant stratum the
matched-pairs conditional MLE has a closed form — the stratum likelihood depends only on the
*difference* of the linear predictor between the two occupants, so fitting reduces to an
intercept-free logistic regression on covariate differences (`her value − his value`) over
**discordant pairs only** (one died, one didn't — concordant pairs carry no information and drop
out, exactly like v0's `double_pair`). Covariates: seat, age difference, and age-difference ×
mean-age (a closed-form quadratic-in-age term, no splines). CIs are cluster-robust
(`cov_type="cluster"` on crash id), cross-checked against the existing crash-clustered bootstrap.

Two new estimand families, kept structurally apart:

- **`fars_condlogit_*`** — the mixed-sex channel (one female, one male occupant per vehicle).
  `sex_effect_frontal_{variant}_band_{lo}-{hi}` is the age-adjusted per-band female odds ratio, the
  direct age-adjusted answer to the trend question; `_pooled` collapses all bands into one
  coefficient; `sex_trend_slope_*` is a separate continuous model-year trend fit (never jointly with
  the band dummies — band is a coarsened function of year, so the two are collinear together);
  `seat_effect_rightfront_*`, `age_slope_*`, and `age_curvature_*` are the shared nuisance
  coefficients. Every `_band_*` row also comes in two sensitivity variants: `_separatenuisance`
  (seat/age refit independently inside that band, instead of pooled across bands) and
  `_agepiecewise` (age entered as piecewise-linear with a knot at 65, instead of quadratic). Neither
  sensitivity variant is headlined — they exist to show whether the default's pooling and
  functional-form choices are load-bearing.
- **`fars_samesex_*`** — a structurally separate channel: **right-front-passenger-vs-driver**
  fatality odds (a value above 1 means the passenger seat is the deadlier one) within discordant
  male-male pairs and, separately, discordant female-female pairs. This is what actually
  identifies the seat × sex interaction: within any mixed-sex pair the female contrast is fixed at 1,
  so a `female × seat` term computed from mixed pairs is algebraically identical to the seat main
  effect — not underpowered, *unidentified*. `seat_effect_male_*` / `seat_effect_female_*` are the
  raw same-sex baselines; `seatsex_interaction_*` is their **log**-ratio (log of the female OR over
  the male OR — its null is 0, not 1), with a bootstrap CI. The `_ageadj` variants of all three
  refit each baseline with a quadratic within-pair age-difference term and are the headline
  versions: the two cohorts differ sharply in within-pair age structure (female-female pairs carry
  a much older right-front passenger far more often), so the raw ORs mix the seat effect with who
  sits where at what age. This channel never shares nuisance parameters with the mixed-sex channel,
  and same-sex pairs re-admit some of the confounding the double-pair design exists to close (who
  rides with whom isn't random), so every interaction number here carries a weaker identifying
  assumption than everything else on this page — the dashboard says so next to the number, not in a
  footnote.

`fars_doublepair_*` (v0) is untouched — same three rows per frontal cut, still headed by
`fars_doublepair_pooled_f_vs_m_seatconfounded_*`, still quarantined from the headline by name alone.
The pooled, age-adjusted `fars_condlogit_sex_effect_frontal_{variant}_pooled` figure is **not** a
drop-in replacement for v0's seat-balanced 4.1%: on FARS 2015–2024 it comes out around **1.18×**
(core, CI 1.11–1.26, clearly excluding 1.0) rather than v0's 4.1% (not significant). The two measure
related but different things — v0 nets out the seat effect via the geometric mean of two seat-split
ratios with no age term; v1 pools an age-adjusted, band-dummy model with a shared seat coefficient
across the whole 2015–2024 span — and this design doesn't resolve which framing is closer to the
true seat-balanced sex effect. Read both, not one instead of the other.

**Power floor.** Any `results` row backed by fewer than 30 discordant pairs is written to the table
(the ledger stays complete) but excluded from headline/dashboard treatment — check
`n_discordant_pairs` in the row's `cohort_def` JSON before trusting a number pulled straight from
SQL. `n_pairs` (eligible) and `n_discordant_pairs` (informative, what the fit actually used) are
tracked separately in every v1 row for exactly this reason.

**The trend question: age adjustment does not settle it — the direction is unidentified on this
window.** The naive unadjusted per-band ratio rises across model-year bands (1970s–90s vehicles
through 2017+), and so does the default age-adjusted
`fars_condlogit_sex_effect_frontal_core_band_*` (1.13 → 1.16 → 1.18 → 1.26). But the
`_separatenuisance` sensitivity fit — the same per-band female OR with seat/age refit inside each
band instead of pooled — runs the other way: **1.26 → 1.21 → 1.18 → 1.10, a monotone decline in the
published direction**. And the data reject the pooled model's own identifying assumption: the
age-adjusted seat effect is not constant across bands (0.89 [0.72–1.10] in the oldest band up to
**1.34 [1.17–1.53]** in the newest — CIs disjoint), and with the female occupant in the right-front
seat in ~74% of pairs in every band, a band-varying seat effect that the pooled model averages away
loads directly onto the per-band female dummies. The continuous `sex_trend_slope_frontal_core`
model (fit with `mod_year` windowed to 1970–2026, the same span the bands cover — FARS's 9999 "not
reported" sentinel and a few implausible pre-1970 model years never reach the fit as leverage
points on what is otherwise a raw numeric regressor) is **+0.044** log-odds per decade (95% CI
−0.026 to +0.115, straddling zero) for core, and inherits the same pooling assumption the band
data reject. The piecewise-linear age variant agrees with the quadratic default to within ~0.1% in
every band, so age functional form isn't what's driving any of this. Net: on 2015–2024 FARS **the
trend direction is unidentified** — it rises under pooled nuisance, falls monotonically under
per-band nuisance, the data reject pooling, and no pairwise band contrast is significant under
either specification. Extending ingestion back to 2000–2014 (thickening the two oldest bands,
which are nearly extinct by 2015) is the scoped follow-up that could actually move this.

One number worth flagging so the page doesn't look self-contradictory: the age-adjusted, pooled
seat effect (`fars_condlogit_seat_effect_rightfront_frontal_core` = 1.05 [0.99–1.12]) is about half
of v0's +10.8% and no longer significant — age adjustment and the band-varying seat effect above
absorb much of what v0's geometric-mean split attributed to the seat — and the head-on cut flips it
clearly protective (0.81 [0.73–0.89]), consistent with v0's head-on reversal.

**The seat × sex interaction: the raw asymmetry is mostly age composition, and what survives
adjustment is fragile.** The raw passenger-vs-driver baselines do run in opposite directions by
sex: among male-male pairs the *driver* dies more often (right-front OR 0.85, CI 0.79–0.92,
n=2,453 discordant pairs); among female-female pairs the *passenger* does (OR 1.61, CI 1.46–1.77,
n=1,788). The raw interaction **log**-ratio is +0.63 (CI +0.51 to +0.76, null = 0) for core — a
1.9× ratio on the multiplicative scale. But the two cohorts sit on radically different age
structures: female-female pairs carry a right-front passenger 11+ years older than the driver in
~39% of discordant pairs (male-male: the passenger averages slightly *younger*), and age — not
seat — is what kills in those pairs. The `_ageadj` rows adjust each baseline for the within-pair
age difference, and roughly 60% of the raw log-ratio evaporates: adjusted male-male 0.90
[0.83–0.98], female-female 1.15 [1.02–1.30], interaction **+0.25 (CI +0.10 to +0.40)** for core.
The residual is statistically significant but rests on the large-age-gap (plausibly parent–child)
pairs where the quadratic age model extrapolates furthest; restricted to age-comparable pairs the
interaction is null. Read it as **an unadjusted asymmetry driven substantially by who sits where
at what age, with a smaller adjusted residual that is not robust evidence of a female-specific
frontal seat penalty** — not as a settled interaction. The same-sex-pair identifying assumption in
§3/§6 of the design notes applies in full on top of all this — same-sex pairs aren't sex-paired
within the same crash, so who rides with whom could correlate with severity or vehicle type in
ways the mixed-pair design rules out by construction.

## Usage

```bash
pip install -e .
crashgap ingest --years 2015-2024    # ~300 MB of zips from static.nhtsa.gov, cached in data/raw
crashgap analyze --years 2015-2024   # pooled + decomposed estimates -> results table + stdout
crashgap dashboard                   # Streamlit tile on :8501
```

Everything lands in `data/crashgap.db`. Re-running either step is idempotent. v0's `analyze` writes
three estimands per frontal cut: the seat-balanced sex effect, the right-front seat effect, and the
pooled ratio explicitly named `_seatconfounded` so it can never be mistaken for the headline. v1
adds the `fars_condlogit_*` and `fars_samesex_*` rows described above, under the same `results`
table and the same `run_ts` per run, and never touches or renumbers v0's rows.

## Refresh

FARS publishes one new year annually (and finalizes the previous ARF release). `deploy/refresh.sh`
re-pulls the two newest years and re-runs the analysis; point a monthly cron at it and the numbers
update themselves when NHTSA ships.

## Roadmap

- **v1** (this release) — the age-adjusted conditional logit and same-sex seat × sex interaction
  described above. It settles *neither* headline question: the model-year trend direction is
  unidentified on this window (it flips with the nuisance-pooling assumption, which the data
  reject), and the interaction that survives age adjustment is fragile — see the honest reads
  above. What v1 does settle is *why* the naive versions of both numbers mislead. Extending
  ingestion to 2000–2014 to thicken the two oldest bands is the natural next step for the trend
  specifically.
- **v2** — CISS/NASS-CDS with MAIS injury grading and survey weights: the rung that reaches the
  severity-adjusted odds ratios the literature reports.

## Data

NHTSA FARS National CSV releases, 1975–2024, US public domain. This repo pools 2015–2024, where the
codings the analysis touches are stable. The 2024 file is an Annual Report File and gets revised by
NHTSA before final release.

## License

MIT. The data is US-federal public domain; the numbers are only as good as their caveats, which
ship on the same page.
