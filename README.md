# CrashGap

Female-vs-male fatality risk in the *same* crash, computed from public-domain NHTSA
[FARS](https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars) data, with
the seat-position confound separated out.

Cars are crash-tested around the median male body. The literature has measured the consequence
for decades, but there was no page that recomputes the number when NHTSA publishes a new year.
CrashGap is that page and the pipeline behind it. Every estimate lands in a SQLite `results`
table with its run timestamp, git commit, FARS years and serialized cohort definition, and the
dashboard leads with the decomposition instead of the headline-friendly number.

## The headline

Take belted front-seat mixed-sex pairs in the same vehicle in a fatal frontal crash and pool
them: women come out 1.09× more likely to be the one who died (FARS 2015–2024, 22,721 pairs).
That number is misleading. Seat position is the one attribute that differs within every pair,
and it is about 73/27 collinear with sex: she is the passenger in 16,656 of those pairs and the
driver in only 6,065. Split by seat configuration and the ratio flips:

| configuration | pairs | she died : he died |
|---|---:|---:|
| she is the passenger, he drives | 16,656 | 1.153 |
| she drives, he is the passenger | 6,065 | 0.940 |

If sex alone drove the outcome those two ratios would agree. Evans' geometric means separate
the effects: with *f* the female multiplier and *s* the right-front multiplier, the two
configurations measure *f·s* and *f/s*, so `sqrt(product)` recovers *f* and `sqrt(quotient)`
recovers *s*:

| effect | frontal core | 95% CI | significant |
|---|---:|---|---|
| being female (seat-balanced) | +4.1% | −1.5% to +10.1% | no |
| sitting right-front | +10.8% | +4.7% to +17.1% | yes |

On FARS 2015–2024, in the primary frontal cut, the seat effect is larger than the sex effect,
and the pooled 1.09 is mostly seat.

The headline stays on the modern decade although the database holds FARS 2000–2024. The
question "does being female raise fatality risk in the same crash today" should not absorb
1990s vehicles as more history is ingested. The full span feeds the trend and same-sex
sections, where the older vehicle bands have enough pairs.

The frontal cuts disagree, and the disagreement carries information. In the strict head-on cut
(`MAN_COLL=2`) the female effect is +11.5% and significant, and the seat effect reverses: the
driver fares worse. That is physically plausible. Head-on collisions are predominantly offset
toward the driver's side, which is why the driver-side small-overlap crash test existed for six
years before a passenger-side one was added. Depending on crash geometry the seat-balanced
female effect sits somewhere between low single digits and low double digits, and ten years of
FARS cannot pin it down further. Quoting only the head-on row, or only the core row, is
cherry-picking.

None of this contradicts the larger published severity-adjusted gap (Bose 2011 reports MAIS 3+
odds of about 1.47). That is a different quantity on different data, measured in the v2 section
below.

## Method

Evans double-pair design: only vehicles carrying exactly one eligible male and one eligible
female (both belted, front outboard, age 16–96, light vehicle) enter. That holds crash
severity, delta-V, vehicle and impact direction physically constant within the vehicle. FARS is
a census of *fatal* crashes, so a raw occupant-level odds ratio would be dominated by
crash-involvement selection and can point the wrong way. Hence the within-vehicle design, and
the seat split on top of it. Frontal is pinned to IMPACT1 ∈ {11, 12, 1} (Forman 2019), with the
wider fan and head-on cut reported as sensitivity variants. Every code set is a literal in
`crashgap/codebook.py` with its source next to it. Confidence intervals are percentile
bootstraps that resample whole crashes, never persons, with both seat strata resampled inside
one replicate.

The finding itself is settled literature (Evans 1988; Bose 2011; Forman 2019; Atwood, Noh &
Craig 2023). This repo adds the live, versioned form and says so on the page. Every crash here
already killed someone, so nothing above is a population rate. Age and severity adjustments
come in v1 and v2 below.

## v1: age adjustment, the trend, and the seat × sex interaction

v0's decomposition cannot separate age from sex and says nothing about the model-year trend or
about whether the seat effect differs by sex. v1 adds a within-vehicle differenced conditional
logit (`statsmodels` is the new dependency; `Logit` is the only call site). For a 2-occupant
stratum the matched-pairs conditional MLE has a closed form: the stratum likelihood depends
only on the difference of the linear predictor between the two occupants, so the fit reduces to
an intercept-free logistic regression on covariate differences (her value minus his) over
discordant pairs only. Concordant pairs carry no information and drop out, as in v0.
Covariates: seat, age difference, and age-difference × mean-age (a closed-form quadratic in
age). CIs are cluster-robust on crash id and cross-checked against the crash-clustered
bootstrap.

Two estimand families, kept structurally apart:

- `fars_condlogit_*` is the mixed-sex channel. `sex_effect_frontal_{variant}_band_{lo}-{hi}` is
  the age-adjusted per-band female odds ratio and the direct answer to the trend question;
  `_pooled` collapses the bands into one coefficient; `sex_trend_slope_*` is a separate
  continuous model-year fit (never jointly with the band dummies, which are a coarsened
  function of the same variable); `seat_effect_rightfront_*`, `age_slope_*` and
  `age_curvature_*` are the shared nuisance coefficients. Every `_band_*` row also exists in
  three sensitivity variants: `_separatenuisance` (seat and age refit inside each band),
  `_agepiecewise` (piecewise-linear age with a knot at 65), and `_vehage12` (vehicles at most
  12 years old at crash, which puts bands on comparable vehicle-age support and makes band and
  calendar period nearly synonymous, the same mixing the published FARS trend estimates
  accept). No sensitivity variant is headlined. A band-level claim that does not hold across
  all of them depends on a modelling assumption rather than on the data.
- `fars_samesex_*` is a structurally separate channel: right-front-passenger-vs-driver fatality
  odds (above 1 means the passenger seat is the deadlier one) within discordant male-male pairs
  and, separately, female-female pairs. This is what identifies the seat × sex interaction.
  Within a mixed-sex pair the female contrast is fixed at 1, so a `female × seat` term computed
  from mixed pairs is algebraically the seat main effect; the interaction is unidentified
  there, at any sample size. `seat_effect_male_*` / `seat_effect_female_*` are the raw
  baselines; `seatsex_interaction_*` is their **log**-ratio (its null is 0, not 1). The
  `_ageadj` variants refit each baseline with a quadratic within-pair age-difference term and
  are the headline versions, because the two cohorts differ sharply in who sits where at what
  age. `seatsex_interaction_ageadj_agecomparable_*` restricts the fit to pairs at most 10 years
  apart, where the age model interpolates instead of extrapolating; quote it next to the
  full-cohort `_ageadj` row. This channel never shares nuisance parameters with the mixed-sex
  channel. Same-sex pairs are not sex-paired within the same crash, so who rides with whom can
  correlate with severity or vehicle type. Every number from this channel rests on a weaker
  identifying assumption than the rest of the page, and the dashboard says so next to the
  number.

`fars_doublepair_*` (v0) is untouched, still headed by the row explicitly named
`_seatconfounded`. The pooled age-adjusted `_pooled` figure is not a drop-in replacement for
v0's seat-balanced +4.1%: on FARS 2015–2024 it reads about 1.18× (core, CI 1.11–1.26). The two
measure related but different things. v0 nets out the seat effect through the geometric mean of
two seat-split ratios with no age term; v1 pools an age-adjusted band model with one shared
seat coefficient. This design does not resolve which is closer to the true seat-balanced
effect, so read both.

Power floor: a `results` row backed by fewer than 30 discordant pairs is written to the table
but never headlined. Check `n_discordant_pairs` in the row's `cohort_def` before trusting a
number pulled straight from SQL; `n_pairs` counts the eligible cohort including concordant
pairs and overstates effective power on its own.

### The trend, on the full 2000–2024 window

The v1 release's 2015–2024 window left the trend unidentified: the per-band pattern flipped
with the nuisance-pooling assumption and the two oldest bands had almost no power (vehicles
from the 1990s are nearly extinct on modern roads). Ingesting FARS 2000–2014 multiplied the
oldest band's discordant pairs by ten (553 → 5,532). Where that leaves the trend question:

- Identified: pre-2000 vehicles carry a clearly higher age-adjusted female penalty than 2000s
  vehicles. `band_1970-1999` = 1.26 [1.19–1.33] against `band_2000-2009` = 1.09 [1.03–1.15],
  with disjoint CIs under every specification (default pooled-nuisance, `_separatenuisance`,
  `_vehage12`, `_agepiecewise`). That reproduces the direction of the published
  Atwood/Noh/Craig decline, in-window, for the first time in this project.
- Not found: any decline after the 2000s band. The 2010–2016 band sits above the 2000s level in
  every specification (1.14–1.18) and the newest band does too (1.23–1.26), except under the
  separate-nuisance refit, where the two are indistinguishable (1.10 vs 1.11). The default fit
  reads the newest band significantly *above* the 2000s band (Wald contrast on the
  pooled-nuisance model's covariance, p ≈ 0.03), but that contrast does not survive the
  separate-nuisance refit, and no post-2000 contrast has disjoint CIs under any specification.
  It stays a spec-dependent hint of a reversal and is not counted as a finding. The newest band
  still flips with the nuisance choice (1.26 pooled vs 1.10 separate; the age-adjusted seat
  effect is about 1.0 in the older bands and 1.34 [1.17–1.53] in 2017+, so the data reject
  pooling exactly where it matters most). The continuous `sex_trend_slope_frontal_core` over
  the full span is −0.025 log-odds per decade (95% CI −0.062 to +0.013), consistent with a slow
  decline and with zero. The published continued shrink to ~5.8% in the newest bands is not
  reproduced. Candidate explanations: the 2020–2024 calendar years this window adds beyond that
  study's 2019 endpoint (a period whose crash mix shifted sharply), or a genuine plateau. Model
  year, vehicle age and calendar period are linearly dependent and none of them varies within a
  vehicle, so this design cannot say which.

The piecewise-linear age variant agrees with the quadratic default to within ~0.2% in every
band, so age functional form is not what drives any of this. The `mod_year` regressor stays
windowed to 1970–2026, so FARS's 9999 "not reported" sentinel never reaches the continuous fit
as a leverage point.

One number to know so the page does not look self-contradictory: the age-adjusted pooled seat
effect (`fars_condlogit_seat_effect_rightfront_frontal_core`) is small and not significant on
either window (1.05 [0.99–1.12] on 2015–2024, 1.03 [0.99–1.07] on 2000–2024), about half of
v0's seat-balanced +10.8% (modern; +8.1% full window). Age adjustment and the band-varying seat
effect absorb much of what v0's geometric-mean split attributed to the seat. The head-on cut
flips it clearly protective (0.81 [0.73–0.89] modern, 0.78 [0.74–0.83] full), consistent with
v0's head-on reversal.

### The interaction, on the full 2000–2024 window

The raw passenger-vs-driver baselines run in opposite directions by sex. Among male-male pairs
the *driver* dies more often (right-front OR 0.89, CI 0.85–0.94, n=7,140 discordant pairs);
among female-female pairs the *passenger* does (OR 1.56, CI 1.47–1.65, n=4,744). The raw
interaction log-ratio is +0.56 (CI +0.48 to +0.63, null = 0) for core. Most of that is age
composition: a female-female pair carries a right-front passenger 11+ years older roughly twice
as often as a male-male pair does, and in those pairs age is the strongest predictor of who
dies. The `_ageadj` rows adjust each baseline and most of the raw log-ratio goes away: adjusted
male-male 0.91 [0.86–0.95], female-female 1.12 [1.04–1.20], interaction +0.21 (CI +0.12 to
+0.30) for core. On the 2015–2024 window alone this residual looked fragile; with 2.8× the
discordant pairs it is statistically solid and stable in direction across calendar eras
(2000–2009: +0.22, 2010–2017: +0.15, 2018–2024: +0.25, two of the three individually
significant). The `_agecomparable` fragility row keeps it short of settled: restricted to pairs
within 10 years of each other, where the quadratic age model interpolates instead of
extrapolating, the interaction reads +0.09 (CI −0.01 to +0.20), roughly half the full-cohort
value with a CI touching zero. Read the two numbers together: consistent evidence of a
female-specific right-front penalty, magnitude somewhere between roughly +10% and +23%, carried
disproportionately by large-age-gap pairs. The same-sex identifying assumption in §3/§6 of the
design notes applies to all of it.

## v2: the severity rung (CISS and NASS-CDS)

Everything above is a *fatality* contrast inside fatal crashes, because FARS carries nothing
else. The numbers the crash-test-dummy debate actually cites (Bose 2011's MAIS 3+ odds ratio
1.47, Craig 2024's MAIS 2+ ~1.75) are severity-adjusted injury odds, measured on
crash-investigation samples with hospital-grade AIS injury grading, reconstructed delta-V and
survey weights. v2 ingests both sources (CISS 2017–2024 CSV releases and NASS-CDS 2000–2015 SAS
files) into a dedicated `sev_occupant` table and fits the survey-convention model: a
design-weighted logistic pseudo-MLE with Taylor-linearized variance (strata = `PSUSTRAT`,
clusters = `PSU`, weights = `CASEWGT`/`RATWGT`) and t-based CIs on the design's own degrees of
freedom, 15 (pooled NASS) to 28 (pooled CISS) here, so the intervals come out visibly wider
than naive ones. Estimand family:
`{ciss,nass}_svylogit_female_or_{mais2plus,mais3plus}_frontal_*`.

Cohort: belted front-outboard adults (16–96) with known sex and graded MAIS in frontal
light-vehicle tow-away crashes; frontal means the primary damage event has plane F at 11–1
o'clock (`ve` GAD1/DOF1 for NASS, the CDC file's ranked events for CISS), symmetric with the
FARS core zone. Three covariate tiers per outcome: `base` (age, seat, model-year band), `dv`
(adds ΔV and ΔV² on the subset where reconstruction succeeded), `dvanthro` (adds height and
BMI). Height and BMI partly mediate a sex effect, so the third tier answers what remains net of
body size; it does not test whether the gap is real. Sensitivities: `_wtrim95` caps weights at
the cohort's 95th percentile (every row's `cohort_def` carries the Kish effective n and the
max/median weight ratio, which makes the Viano 2025 weight-instability critique measurable),
and `_ais08` refits NASS on its dual-coded AIS2008 grading, sizing the revision boundary that
sits between the Bose-era and Craig-era benchmarks. The power floor here is 30 unweighted
outcome events.

Where the first canonical run landed (CISS 52,943 + NASS-CDS 150,897 ingested occupants;
frontal cohorts 13,145 / 28,220): the delta-V-adjusted estimates sit where the literature sits.
NASS MAIS 3+ `dv` = 1.50 [1.10–2.04] against Bose's published 1.47; CISS MAIS 2+ `dv` = 1.91
[1.23–2.96] against Craig's ~1.75. The base tiers sit below them, because women's frontal
crashes carry lower delta-V on average, the direction every published severity analysis
reports. The body-size tier does not absorb the effect: NASS MAIS 3+ stays at 1.51, both
MAIS 2+ cells rise, and only the underpowered CISS MAIS 3+ cell falls (not significant in
either tier). The measured fragilities ship next to those numbers. Single cases carry weights
up to ~500× the median (Kish effective n ~10% of nominal; the `_wtrim95` refit moves NASS
MAIS 2+ from 1.72 to 1.45 without changing its significance). Grading the *same* dual-coded
2010–2015 cohort both ways isolates the AIS revision itself: MAIS 3+ reads 1.31 [0.62–2.76]
under AIS90 and 0.93 [0.60–1.43] under AIS2008, while MAIS 2+ barely moves (1.93 vs 1.95). Part
of any Bose-vs-Craig gap at the serious-injury threshold comes from injury coding rather than
crash physics.

v2 never pools CISS with NASS (different designs, eras, AIS revisions), never pools either with
FARS (different estimand entirely), and does not claim a reproduction of Bose or Craig: frontal
definitions, cohorts and covariate sets differ in ways `docs/v2-design.md` documents. The
dashboard shows the published numbers for orientation, with the differences beside them.

## Usage

```bash
pip install -e .
crashgap ingest --years 2000-2024      # ~470 MB of zips from static.nhtsa.gov, cached in data/raw
crashgap analyze --years 2015-2024     # modern-window estimates (the dashboard headline)
crashgap analyze --years 2000-2024     # full-window estimates (the trend sections)
crashgap analyze-v1 --years 2015-2024  # conditional-logit rows, modern window
crashgap analyze-v1 --years 2000-2024  # conditional-logit rows, full window
crashgap ingest-severity               # CISS + NASS-CDS for the v2 severity rung
crashgap analyze-v2                    # severity estimands
crashgap dashboard                     # Streamlit dashboard on :8501
```

Everything lands in `data/crashgap.db` and re-running any step is idempotent. The dashboard
reads the latest run per (estimand family, window), so both windows need both `analyze` and
`analyze-v1` runs. v0's pooled ratio is explicitly named `_seatconfounded` so it cannot be
mistaken for the headline; v1 and v2 write their own estimand families under the same `results`
table and never touch v0's rows.

## Refresh

FARS publishes one new year annually and finalizes the previous release; CISS publishes one new
year annually (NASS-CDS ended 2015 and never changes). `deploy/refresh.sh` re-pulls the newest
years and re-runs every analysis pass. Point a monthly cron at it and the numbers update
themselves when NHTSA ships.

## Data

NHTSA FARS National CSV releases, 1975–2024, US public domain; this repo pools 2000–2024. Two
FARS recodes sit inside that span and were verified against the real files before widening it
(see `crashgap/codebook.py`): the unknown-age sentinel changed schemes in 2009 (99 → 998/999,
both outside the 16–96 age window), and IMPACT1 gained side-specific codes in 2010, which
drains the *wide* frontal variant's clock points 2 and 10 while the core zone {11, 12, 1} and
the MAN_COLL=2 head-on cut stay continuous across the boundary. The 2024 file is an Annual
Report File and gets revised by NHTSA before final release. CISS and NASS-CDS are US public
domain as well.

## License

MIT. The data is US-federal public domain; the caveats ship on the same page as the numbers.
