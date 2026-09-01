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

(The headline deliberately stays on the modern decade even though the database now holds FARS
2000–2024: "does being female raise fatality risk in the same crash *today*" should not quietly
absorb 1990s vehicles as history is ingested. The full span exists for the model-year trend and
same-sex sections below, where the older vehicle bands actually have power.)

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
severity-adjusted odds ratios, age-adjusted (women in these pairs average ~1.3 years younger), or
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
  coefficients. Every `_band_*` row also comes in three sensitivity variants: `_separatenuisance`
  (seat/age refit independently inside that band, instead of pooled across bands),
  `_agepiecewise` (age entered as piecewise-linear with a knot at 65, instead of quadratic), and
  `_vehage12` (vehicles at most 12 years old at crash, putting the bands on comparable
  vehicle-age support at the price of making band and calendar period nearly synonymous — the
  same mixing the published FARS trend estimates accept). No sensitivity variant is headlined —
  they exist to show whether the default's pooling, functional-form and period-mixing choices are
  load-bearing, and a band-level claim that doesn't hold across all of them is treated as a claim
  about an assumption, not about the data.
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
  sits where at what age. `seatsex_interaction_ageadj_agecomparable_*` is the fragility check on
  the adjusted headline: the same fit restricted to pairs with a within-pair age gap of at most
  10 years, where the quadratic adjustment interpolates instead of extrapolating — quote it next
  to the full-cohort `_ageadj` row, never one without the other. This channel never shares nuisance parameters with the mixed-sex channel,
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

**The trend question, on the full 2000–2024 window: the decline into the 2000s is real; a
continued decline into the newest vehicles is not found.** This was the scoped follow-up from the
v1 release, whose 2015–2024 window left the trend direction unidentified (the per-band pattern
flipped with the nuisance-pooling assumption, and the two oldest bands had almost no power —
vehicles from the 1990s are nearly extinct on modern roads). Ingesting FARS 2000–2014 multiplied
the oldest band's discordant pairs by ten (553 → 5,532), and the picture that emerges is
two-sided:

- **What is now identified:** pre-2000 vehicles carry a clearly higher age-adjusted female
  penalty than 2000s vehicles — `fars_condlogit_sex_effect_frontal_core_band_1970-1999` = 1.26
  [1.19–1.33] against `band_2000-2009` = 1.09 [1.03–1.15], with disjoint CIs under **every**
  specification (default pooled-nuisance, `_separatenuisance`, `_vehage12`, `_agepiecewise`).
  That reproduces the *direction* of the published Atwood/Noh/Craig decline, in-window, for the
  first time in this project.
- **What is still not found:** any decline after the 2000s band. The 2010–2016 band sits above
  the 2000s level in every specification (1.14–1.18), and the newest band does too (1.23–1.26)
  except under the separate-nuisance refit, where the two are indistinguishable (1.10 vs 1.11).
  The default fit even reads the newest band significantly *above* the 2000s band (Wald
  contrast on the pooled-nuisance model's covariance, p ≈ 0.03) — but that contrast does not
  survive the separate-nuisance refit and no post-2000 contrast has disjoint CIs under any
  specification, so it is a spec-dependent hint of a reversal, not a finding. The newest band
  still flips with the nuisance choice (1.26 pooled vs 1.10
  separate — the age-adjusted seat effect varies by band, roughly 1.0 in the older bands vs
  **1.34 [1.17–1.53]** in 2017+, so the data reject pooling exactly where it matters most). The
  continuous `sex_trend_slope_frontal_core` over the full span is **−0.025** log-odds per decade
  (95% CI −0.062 to +0.013): consistent with a slow decline and with zero. The published
  continued shrink to ~5.8% in the newest bands is *not* reproduced — which may reflect the
  2020–2024 calendar years this window adds beyond that study's 2019 endpoint (a period whose
  crash mix shifted sharply), or a genuine plateau; model year, vehicle age and calendar period
  are linearly dependent and none of them varies within a vehicle, so this design cannot say
  which.

The piecewise-linear age variant agrees with the quadratic default to within ~0.2% in every band,
so age functional form isn't what's driving any of this; and the `mod_year` regressor stays
windowed to 1970–2026, so FARS's 9999 "not reported" sentinel never reaches the continuous fit as
a leverage point.

One number worth flagging so the page doesn't look self-contradictory: the age-adjusted, pooled
seat effect (`fars_condlogit_seat_effect_rightfront_frontal_core`) is small and not significant
on either window — **1.05 [0.99–1.12] on 2015–2024, 1.03 [0.99–1.07] on 2000–2024** — about half
of v0's seat-balanced +10.8% (modern; +8.1% full window). Age adjustment and the band-varying
seat effect above absorb much of what v0's geometric-mean split attributed to the seat — and the
head-on cut flips it clearly protective (0.81 [0.73–0.89] modern, 0.78 [0.74–0.83] full),
consistent with v0's head-on reversal.

**The seat × sex interaction, on the full 2000–2024 window: the raw asymmetry is mostly age
composition; what survives adjustment is now significant and era-stable, but its size still
depends on the age model.** The raw passenger-vs-driver baselines run in opposite directions by
sex: among male-male pairs the *driver* dies more often (right-front OR 0.89, CI 0.85–0.94,
n=7,140 discordant pairs); among female-female pairs the *passenger* does (OR 1.56, CI 1.47–1.65,
n=4,744). The raw interaction **log**-ratio is +0.56 (CI +0.48 to +0.63, null = 0) for core. But
the two cohorts sit on very different age structures (a female-female pair carries a right-front
passenger 11+ years older roughly twice as often as a male-male pair does), and age — not seat —
is what kills in those pairs. The `_ageadj` rows adjust each baseline for the within-pair age
difference and most of the raw log-ratio evaporates: adjusted male-male 0.91 [0.86–0.95],
female-female 1.12 [1.04–1.20], interaction **+0.21 (CI +0.12 to +0.30)** for core. On the v1
release's 2015–2024 window alone this residual looked fragile; with 2.8× the discordant pairs it
is statistically solid and stable in direction across calendar eras (2000–2009: +0.22, 2010–2017:
+0.15, 2018–2024: +0.25 — two of the three individually significant; rerun via
`age_comparable`/`seatsex_interaction_ageadj` on era-filtered frames). What keeps it short of
settled is the `_agecomparable` fragility row: restricted to pairs within 10 years of each other —
where the quadratic age model interpolates instead of extrapolating — the interaction reads
**+0.09 (CI −0.01 to +0.20)**, roughly half the full-cohort value with a CI touching zero. The
honest read is **consistent evidence of a female-specific right-front penalty of modest size,
with the magnitude uncertain between roughly +10% and +23% and carried disproportionately by
large-age-gap pairs**. The same-sex-pair identifying assumption in §3/§6 of the design notes
applies in full on top of all this — same-sex pairs aren't sex-paired within the same crash, so
who rides with whom could correlate with severity or vehicle type in ways the mixed-pair design
rules out by construction.

## v2: the severity rung — CISS and NASS-CDS

Everything above is a *fatality* contrast inside fatal crashes, because that is all FARS can
carry. The published numbers the crash-test-dummy debate actually cites — Bose 2011's MAIS 3+
odds ratio 1.47, Craig 2024's MAIS 2+ ~1.75 — are **severity-adjusted injury odds**, measured on
crash-investigation samples with hospital-grade AIS injury grading, reconstructed delta-V and
survey weights. v2 ingests both of those sources (CISS 2017–2024 CSV releases and NASS-CDS
2000–2015 SAS files) into a dedicated `sev_occupant` table and fits the survey-convention model:
a design-weighted logistic pseudo-MLE with Taylor-linearized variance (strata = `PSUSTRAT`,
clusters = `PSU`, weights = `CASEWGT`/`RATWGT`) and t-based CIs on the design's own degrees of
freedom — 15 (pooled NASS) to 28 (pooled CISS) here, so honest intervals are visibly wider than
naive ones. Estimand family: `{ciss,nass}_svylogit_female_or_{mais2plus,mais3plus}_frontal_*`.

Cohort: belted front-outboard adults (16–96) with known sex and graded MAIS in frontal
light-vehicle tow-away crashes — frontal meaning the primary damage event has plane F at 11–1
o'clock (`ve` GAD1/DOF1 for NASS, the CDC file's ranked events for CISS), symmetric with the
FARS core zone. Three covariate tiers per outcome: `base` (age, seat, model-year band), `dv`
(adds ΔV and ΔV², on the reconstruction-succeeded subset), `dvanthro` (adds height and BMI —
the "is it just body size" tier; height/BMI partly *mediate* a sex effect, so it answers "net of
body size", never "is the gap real"). Sensitivities: `_wtrim95` caps weights at the cohort's
95th percentile (the Viano 2025 weight-instability critique, made measurable — every row's
`cohort_def` carries the Kish effective n and the max/median weight ratio), and `_ais08` refits
NASS on its dual-coded AIS2008 grading, sizing the revision boundary that sits between the
Bose-era and Craig-era benchmarks. The power floor here is 30 unweighted outcome events.

**Where the first canonical run landed (CISS 52,943 + NASS-CDS 150,897 ingested occupants;
frontal cohorts 13,145 / 28,220):** the delta-V-adjusted estimates sit where the literature
sits — NASS MAIS 3+ `dv` = **1.50 [1.10–2.04]** against Bose's published 1.47, CISS MAIS 2+
`dv` = **1.91 [1.23–2.96]** against Craig's ~1.75 — and the base tiers sit below them, because
women's frontal crashes carry lower delta-V on average, the same direction every published
severity analysis reports. The body-size tier does not absorb the effect (NASS MAIS 3+
unchanged at 1.51; both MAIS 2+ cells rise; only the underpowered CISS MAIS 3+ cell falls, and
it is not significant in either tier). The measured fragilities ship next to those numbers:
single cases carry weights up to ~500× the median (Kish effective n ~10% of nominal; the
`_wtrim95` refit moves NASS MAIS 2+ from 1.72 to 1.45 without changing its significance), and
grading the **same** dual-coded 2010–2015 cohort both ways puts the AIS revision's own effect
on the table: MAIS 3+ reads 1.31 [0.62–2.76] under AIS90 and 0.93 [0.60–1.43] under AIS2008
(MAIS 2+ barely moves, 1.93 vs 1.95) — so part of any Bose-vs-Craig gap at the serious-injury
threshold is injury *coding*, not crash physics.

What v2 deliberately does not do: pool CISS with NASS (different designs, eras, AIS revisions),
pool either with FARS (different estimand entirely), claim a reproduction of Bose or Craig
(frontal definitions, cohorts and covariate sets differ in documented ways — the dashboard shows
the published numbers as orientation, with the differences beside them), or grade its own
homework on causality. Design and every deviation from the original v2 plan: `docs/v2-design.md`.

## Usage

```bash
pip install -e .
crashgap ingest --years 2000-2024      # ~470 MB of zips from static.nhtsa.gov, cached in data/raw
crashgap analyze --years 2015-2024     # modern-window estimates (the dashboard headline)
crashgap analyze --years 2000-2024     # full-window estimates (the trend sections)
crashgap analyze-v1 --years 2015-2024  # conditional-logit rows, modern window
crashgap analyze-v1 --years 2000-2024  # conditional-logit rows, full window
crashgap dashboard                     # Streamlit tile on :8501
```

Everything lands in `data/crashgap.db`. Re-running any step is idempotent. The dashboard reads the
latest run per (estimand family, window) — the modern decade feeds the headline sections, the full
span feeds the trend and same-sex sections — so both windows need both `analyze` and `analyze-v1`
runs. v0's `analyze` writes three estimands per frontal cut: the seat-balanced sex effect, the
right-front seat effect, and the pooled ratio explicitly named `_seatconfounded` so it can never be
mistaken for the headline. v1 adds the `fars_condlogit_*` and `fars_samesex_*` rows described
above, under the same `results` table and the same `run_ts` per run, and never touches or renumbers
v0's rows.

## Refresh

FARS publishes one new year annually (and finalizes the previous ARF release). `deploy/refresh.sh`
re-pulls the two newest years and re-runs all four analysis passes (both windows × both estimand
families); point a monthly cron at it and the numbers update themselves when NHTSA ships.

## Roadmap

- **v1** — the age-adjusted conditional logit and same-sex seat × sex interaction described above,
  plus the 2000–2014 ingestion follow-up that put the trend question on a full 25-year window.
  Where that landed: the decline into the 2000s is identified, a continued decline is not found,
  and the adjusted interaction is significant and era-stable but age-model-dependent in size — see
  the honest reads above.
- **v2** (this release) — CISS/NASS-CDS with MAIS injury grading and survey weights: the rung
  that reaches the severity-adjusted odds ratios the literature reports. See the v2 section
  above for where the first run landed; `crashgap ingest-severity` + `crashgap analyze-v2` are
  the commands, and the monthly refresh keeps CISS current as new years publish.

## Data

NHTSA FARS National CSV releases, 1975–2024, US public domain. This repo pools 2000–2024. Two FARS
recodes sit inside that span and were verified against the real files before widening it (see
`crashgap/codebook.py`): the unknown-age sentinel changed schemes in 2009 (99 → 998/999; both fall
outside the 16–96 age window), and IMPACT1 gained side-specific codes in 2010 (which drains the
*wide* frontal variant's clock points 2 and 10 — the core zone {11, 12, 1} and the MAN_COLL=2
head-on cut are continuous across the boundary). The 2024 file is an Annual Report File and gets
revised by NHTSA before final release.

## License

MIT. The data is US-federal public domain; the numbers are only as good as their caveats, which
ship on the same page.
