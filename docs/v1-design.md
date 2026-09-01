# CrashGap v1 Statistical Design (Final)

## 1. Decision summary

v1 keeps v0's core identification move — every comparison is **within the same vehicle**, so
delta-V, striking/struck role, and crash-involvement selection are held constant by construction —
and replaces the manual Evans geometric-mean split with a **within-vehicle differenced conditional
logit** fit via `statsmodels.Logit`. For a 2-occupant vehicle stratum, the true conditional MLE has
a closed form: the stratum likelihood depends only on the *difference* of the linear predictor
between the two occupants, so fitting reduces to an intercept-free ordinary logistic regression on
covariate differences over **discordant pairs only** (concordant pairs carry zero information and
drop out — exactly what v0's `double_pair`/`decompose` already exploit, now stated as a
differenced-covariate logit). No third-party conditional-logit package is needed.

The design fixes two problems v0 could not address:

1. **The trend question** (publish-blocker): the naive unadjusted model-year-band trend runs
   upward, opposite the published shrinking pattern. Age enters the regression explicitly (as a
   within-pair difference, quadratic in form), so the model can test whether age composition
   explains the discrepancy — directly, via a planted band-correlated age confound, not just by
   fitting age and hoping.
2. **Seat × sex interaction**: mixed-sex pairs give exactly two data points (she-drives ratio,
   she-passenger ratio) for exactly two unknowns (sex effect, seat effect) — that system is
   *saturated by construction*. Any attempt to recover a female-by-seat interaction from mixed-sex
   pairs alone is either literally unidentified (a `female × seat` term is algebraically collinear
   with the seat main effect, because the female contrast is fixed at 1 within every mixed
   stratum) or, if it appears identified via an occupant-labeling trick, is silently riding on a
   spurious correlation between that labeling and actual seat position (FARS conventionally
   assigns occupant 1 to the driver, so a labeling-based "interaction" term collapses toward the
   seat main effect). The design routes this test through a **separate same-sex-pair channel**
   instead: driver-vs-passenger fatality odds within male-male pairs and, separately, within
   female-female pairs, discordant on death. The ratio of those two same-sex seat effects is the
   actual, identified interaction test.

Two robustness moves are added on top of the default pooled-nuisance band fit, because pooling
`β_seat`/`β_age` across model-year bands to buy power is an *assumption*, not a fact:

- A **per-band-independent-nuisance** sensitivity fit (seat/age refit separately inside each band)
  bounds how much the trend answer depends on cross-band homogeneity of those nuisance parameters.
- A **continuous model-year trend** model (linear-in-year female effect) is fit as a *separate*
  two-term model — never jointly with the discrete band dummies, since band is a coarsened function
  of year and the two are highly collinear if forced into one design matrix — so the trend
  conclusion isn't an artifact of where the band boundaries fall.

A **quadratic-vs-piecewise-linear age-form sensitivity check** is run before the trend finding is
reported as settled, since the closed-form quadratic-via-differencing trick is a specific
functional-form choice, not a neutral one.

Every run writes rows to the existing append-only `results` table (estimand/point/ci/n/cohort_def/
git_commit), stamped under one `run_ts`, alongside v0's 9 rows unchanged. No point estimate is
headlined on the dashboard with fewer than 30 discordant pairs behind it.

## 2. Model specification

**Unit of estimation**: one row per *discordant* mixed-sex vehicle pair (exactly one of the two
occupants died). Concordant pairs (both survived, or both died) contribute no information to a
matched-pairs conditional likelihood and are dropped, exactly as in v0.

**Outcome**: `y = 1{female occupant died}`.

**Covariates**, each defined as *her value minus his value* within the pair:

- `x_seat = her_is_right_front − his_is_right_front` ∈ {+1, −1}. Coefficient `β_seat`.
- `x_age = age_f − age_m`; `x_age_curv = x_age · mean(age_f, age_m)`. Together these reproduce a
  full quadratic-in-age model (`age + age²`) without differencing squared ages directly:
  `age_f² − age_m² = x_age · (age_f + age_m)`, so `x_age_curv` captures exactly that curvature term
  with two parameters and no splines.
- **No standalone `female` covariate.** Within any discordant mixed-sex pair, her indicator minus
  his is identically 1 — it would be collinear with the intercept, which is absorbed by
  stratification. The female effect is instead carried by design-dependent columns:
  - **Default (trend) model**: one column per model-year band (`x_female_band_2000-2009`, etc.),
    each = 1 iff the pair falls in that band, 0 otherwise. Bands partition the data, so exactly one
    such column is 1 per row — this recovers a separate female coefficient per band. `β_seat` and
    the two age coefficients are shared (pooled) across bands.
  - **Continuous-trend model** (fit separately, never jointly with the band dummies): `x_const = 1`
    for every row (coefficient = female effect at the mean model-year) plus
    `x_trend = (mod_year − mean_mod_year) / 10` (coefficient = log-odds trend per decade). `β_seat`
    and age coefficients shared across the whole span, as in the default model.
  - **Per-band-independent-nuisance model** (sensitivity check): the default band-dummy structure,
    but fit separately within each band's subset of rows, so `β_seat`/age are *not* shared across
    bands. Compared against the default to bound sensitivity to the pooling assumption.

**Fit**: `sm.Logit(y, X).fit()`, no intercept (absorbed by the band/constant column already in X).
Standard errors: `cov_type="cluster"`, `cov_kwds={"groups": crash_id}` for cluster-robust SEs on
`(year, st_case)`, cross-checked against the repo's existing crash-clustered percentile bootstrap
(resample crashes, refit per replicate) as a method-consistency check.

**Age functional form — stated as an assumption, not a neutral choice.** The quadratic-via-
differencing trick is specific: real age-mortality curves can have thresholds (elevated risk past
roughly 65–75) that a quadratic will not capture, and true splines aren't tractable in this
closed-form 2-per-stratum reduction without moving to a full N-per-stratum conditional likelihood,
which nothing here currently needs. Before the trend finding is reported as settled, a **piecewise-
linear age sensitivity variant** (age difference split at a fixed knot, e.g. 65) is fit alongside
the quadratic default and the two are compared qualitatively; a materially different trend
conclusion between the two forms is reported as an open risk, not resolved by picking one.

**Epistemic caveat on "replication."** This design is the standard way to add covariates to a 1:1
matched-pairs logit (the matched-pairs conditional-likelihood identity is textbook — Breslow & Day
1980; Hosmer–Lemeshow ch. 7), and it nests v0's double-pair ratio as the special case with no age
variation. It is **not verified against the exact regression form used in the published
model-year-band literature** this project benchmarks against — that paper may use per-band
Mantel–Haenszel or a GLM on aggregated cells instead. Treat this as *this project's* model, built to
answer the trend and interaction questions on this data, not as a checked reproduction of anyone
else's equations.

## 3. Cohort definition

**Mixed-sex channel (trend + pooled sex/seat estimands).** Occupant-grain eligibility, identical to
the existing pair-eligibility predicate: `is_occupant=1 AND is_front_outboard=1 AND is_belted=1 AND
{is_frontal | is_frontal_wide | is_head_on} AND is_light_vehicle=1 AND age_in_range=1 AND
is_female IS NOT NULL AND died IS NOT NULL`, grouped by `(year, st_case, veh_no)`, restricted to
vehicles with exactly one eligible female and exactly one eligible male occupant. Row grain is
per-occupant (not the aggregated per-vehicle row v0 used), carrying `age`, `mod_year` (banded via
the existing band table), and seat position per occupant. Filtered to discordant pairs
(`SUM(died) = 1`) before fitting.

**Same-sex channel (seat × sex interaction only).** Same occupant-level eligibility predicate,
grouped the same way, but restricted to vehicles with exactly two eligible occupants of the *same*
sex (`SUM(is_female) = 2` or `SUM(is_female) = 0`), discordant on death, one driver one right-front
passenger. This channel is structurally separate from the mixed-sex channel: it is never used to
estimate `β_seat`/`β_age` in the mixed-pair model, and the mixed-pair model never uses same-sex
rows. No pooling of nuisance parameters across the two channels — pooling them would be circular
for exactly the parameter the interaction test is trying to identify.

**Identifying assumption for the same-sex channel, stated plainly**: same-sex pairs are assumed
drawn from a crash-severity/impact-direction mix comparable to mixed-sex pairs. This is weaker than
the assumption underlying the mixed-pair design itself — who rides with whom is not random, so
same-sex pairs re-admit some of the confounding the double-pair design exists to close. It is
partially controlled by keeping the identical frontal/belted/light-vehicle/age-range restriction,
but it is not eliminated, and the dashboard must say so next to the interaction number, not in a
footnote.

## 4. Estimands (results-table rows)

All new estimands are additive — v0's 9 rows (`fars_doublepair_*`) are written unchanged. New rows
use the `fars_condlogit_*` / `fars_samesex_*` prefixes so they can never be mistaken for v0's
seat-confounded pooled ratio.

| Estimand | Definition | Role |
|---|---|---|
| `fars_condlogit_sex_effect_frontal_{variant}_band_{lo}-{hi}` | Per-band female OR, `β_seat`/age pooled across bands (default model) | **The trend answer.** One row per `MOD_YEAR_BANDS` entry. |
| `fars_condlogit_sex_effect_frontal_{variant}_band_{lo}-{hi}_separatenuisance` | Per-band female OR, `β_seat`/age fit independently within that band | Sensitivity check on the pooling assumption. Not headlined. |
| `fars_condlogit_sex_effect_frontal_{variant}_band_{lo}-{hi}_agepiecewise` | Per-band female OR, age entered as piecewise-linear (knot at 65) instead of quadratic | Functional-form sensitivity check. Not headlined. |
| `fars_condlogit_sex_trend_slope_frontal_{variant}` | `β_trend` from the continuous model-year model (log-odds per decade), fit separately from the band-dummy model | Robustness check that the trend conclusion isn't an artifact of band boundaries. |
| `fars_condlogit_sex_effect_frontal_{variant}_pooled` | Single band-free female coefficient, all bands pooled | Validation target only: must equal v0's `decompose().sex_effect` when age is held constant (see Test plan §7.1). |
| `fars_condlogit_seat_effect_rightfront_frontal_{variant}` | `exp(β_seat)`, pooled, age-adjusted | Regression analogue of v0's `seat_effect`. |
| `fars_condlogit_age_slope_frontal_{variant}` | `exp(β_age)` | Diagnostic, not headline. |
| `fars_condlogit_age_curvature_frontal_{variant}` | `exp(β_age_curv)` | Diagnostic, not headline. |
| `fars_samesex_seat_effect_male_frontal_{variant}` | Right-front-passenger-vs-driver fatality OR (pax_only / drv_only; >1 means the passenger seat is deadlier), male-male discordant pairs | Standalone diagnostic: the same-sex seat baseline, reported independently so the interaction ratio below can be eyeballed against a sane baseline. |
| `fars_samesex_seat_effect_female_frontal_{variant}` | Same orientation (passenger-vs-driver), female-female discordant pairs | Same, female-female. |
| `fars_samesex_seatsex_interaction_frontal_{variant}` | Log-ratio of the two rows above, bootstrap CI | **The seat × sex interaction answer.** Named distinctly from `seatconfounded`/`seat_effect_rightfront` so it is never mistaken for the (unidentified, from mixed pairs) interaction. |

Every row's `cohort_def` JSON carries `n_pairs` (eligible) and `n_discordant_pairs` (informative)
separately — discordant-only fitting is correct for the OR, but a reported `n_pairs` that doesn't
distinguish the two overstates effective power, the same trap v0's `DoublePair.n_pairs` already
avoided by exposing both.

## 5. How the trend question is answered

`mod_year` is constant within a pair (both occupants ride in the same vehicle), so it cannot enter
as a differenced covariate — her `mod_year` minus his is identically zero. It enters only through
its interaction with the sex contrast: band-partitioned female dummies (default model) or a
continuous female × trend term (robustness model), per §2.

The design directly tests, rather than assumes, whether age-adjustment reconciles the naive upward
band trend with the published shrinking one:

1. Fit the default pooled-nuisance band model → per-band female OR with bootstrap/cluster-robust
   CI, age-adjusted.
2. Compare against the existing naive unadjusted per-band ratio already in the codebase.
3. Cross-check with the per-band-independent-nuisance model (does the trend survive when `β_seat`/
   age are allowed to vary by band?) and the continuous-trend model (does a single linear slope
   agree in direction and rough magnitude with the discrete per-band pattern?).
4. Cross-check with the piecewise-linear age form (does the trend conclusion depend on the
   quadratic assumption?).

**Identifying assumption, stated as a single explicit, testable claim**: in the default model,
`β_seat` and the age curve are constant across model-year bands. If false (e.g. differential
seatbelt/airbag improvement by seat position over eras), the pooled nuisance fit contaminates the
per-band female coefficients — which is exactly why the per-band-independent-nuisance model is run
alongside it, not added later as an afterthought.

Given v0's finding that ~800–900 she-drives pairs exist per direction across the whole 2015–2024
window (so roughly 200 per band before splitting further), and that v0's *unadjusted* band CIs
already overlap fully, the honest deliverable from this design is "consistent with a shrinking
trend after age adjustment" or "cannot rule out flat" — not a confirmed monotone point-estimate
trend. Any per-band cell with fewer than 30 discordant pairs is reported in the raw table but
excluded from headline dashboard treatment (greyed out / marked low-power).

## 6. How seat × sex is tested

**Why mixed pairs cannot answer this, stated algebraically.** Within any mixed-sex stratum the
female contrast is fixed: `x_female ≡ 1` (her indicator minus his). A `female × seat` interaction
term would therefore be exactly `x_female · x_seat ≡ x_seat` — algebraically identical to the seat
main effect itself. This is not a power problem or a "not significant" result; it is a system with
exactly two observable moments (she-drives ratio, she-passenger ratio) for exactly two unknowns
(sex effect, seat effect), saturated by construction. No amount of additional mixed-pair data, and
no relabeling trick applied to mixed-pair rows, manufactures a third independent moment — and a
labeling-based version of the trick (e.g. assigning occupant "A"/"B" by a record-order field) is
additionally vulnerable to collapsing toward the seat main effect if that field correlates with
actual seat position, which it plausibly does under FARS's convention of numbering the driver
first.

**The fix: a genuinely separate, identified channel.** Same-sex discordant pairs (§3) give a
driver-vs-passenger fatality OR with the sex contrast structurally absent — a physical baseline for
"how much worse is the right-front seat in a frontal impact," computed once within male-male pairs
and once within female-female pairs. The interaction estimand is the log-ratio of those two
numbers, `fars_samesex_seatsex_interaction_frontal_{variant}`, with its own bootstrap CI (resample
crashes within each same-sex cohort, refit, take the percentile CI of the log-ratio).

This channel answers a related but not identical question to the strongest crash-test-dummy claim
in the literature (general driver/passenger fatality asymmetry by sex, versus frontal-specific
restraint/airbag failure specifically for women) and is the most power-starved part of the whole
design — two-occupant fatal crashes skew heavily mixed-sex (family/couples), so eligible
male-male, and especially female-female, discordant pairs are a small fraction of the mixed-pair n.
Expect this design to make the interaction **testable and honestly null-or-wide**, not to resolve
the crash-test-dummy claim outright — a null result here is not strong evidence of no effect, and
the dashboard must say so next to the number, not below the fold.

## 7. Test plan (planted-effect tests)

All planted tests extend `tests/conftest.py`'s `add_pairs` pattern with an `age` override per
occupant (currently absent) and a new same-sex-pair helper, mirroring `add_pairs`'s shape.

**7.1 Exact-equivalence test (regression, against v0).**
Plant mixed-sex pairs with `age_diff ≡ 0` for every pair (identical ages within each pair) across
the existing seat/who-died cells. `fars_condlogit_sex_effect_frontal_{variant}_pooled`'s fitted
female OR must equal v0's `decompose().sex_effect` to tight tolerance (`math.isclose`,
`abs_tol=1e-6`). Proves the differenced-logit reduces correctly to Evans' geometric mean absent age
variation — the load-bearing check that this isn't a silently different quantity from v0's headline
number.

**7.2 Planted age effect, no sex/seat effect.**
Synthetic occupants with `died` generated from a known logistic function of age
(`logit(p) = -3 + 0.04 * age`), true sex effect = 1.0, true seat effect = 1.0. Assert:
(a) `β_age`/`β_age_curv` recover the planted slope/curvature within their bootstrap CI;
(b) `fars_condlogit_sex_effect_frontal_{variant}_pooled` ≈ 1.0.

**7.3 Planted band-varying sex effect (no age confound).**
Three model-year bands with distinct true female ORs — 3.0 / 1.5 / 1.0 — and age balanced within
pairs in every band (no age-sex correlation). Assert per-band point estimates track the planted
values within CI and are **not** smeared toward a common mean by the pooled-nuisance fit (compare
against the per-band-independent-nuisance model, which should also recover them).

**7.4 The trend-question test (repairs the identified gap — direct proof age-adjustment can
reconcile a spurious naive trend).**
Plant a **flat true female effect (OR = 1.0 in every band)**, but with a female-vs-male age gap
that differs by band and where younger age is protective:

- Band 1 (oldest vehicles): women 8 years younger than their paired men on average.
- Band 2: women 4 years younger.
- Band 3 (newest vehicles): no age gap.
- Age effect: `logit(p) = -3 + 0.05 * age` (older = more likely to die), same for both sexes.

Assert:
(a) the **naive, unadjusted** per-band ratio shows a spurious *increasing* trend across bands
(mirroring the real naive-data pattern), because the age-gap advantage shrinks band to band;
(b) the **age-adjusted** `fars_condlogit_sex_effect_frontal_{variant}_band_{lo}-{hi}` recovers
≈1.0 (flat) in every band, within CI.

This is the mandatory template for the publish-blocker: it is the direct, targeted proof that
age-adjustment *can* remove a spurious trend caused by band-correlated age drift, not just a
generic "age coefficient recovers" check.

**7.5 Same-sex seat×sex interaction — null and true cases.**
Same-sex synthetic pairs, discordant on death:
- *Null case*: plant equal driver/passenger OR in both cohorts — male-male OR = 1.3, female-female
  OR = 1.3. Assert `fars_samesex_seatsex_interaction_frontal_{variant}` ≈ 0 (log-ratio) within CI.
- *True-effect case*: plant female-female driver/passenger OR = 2.0, male-male OR = 1.2. Assert the
  interaction estimand recovers `log(2.0 / 1.2)` within CI.

**7.6 Continuous-trend model cross-check.**
Plant a linear log-odds trend in the female effect across `mod_year` (e.g. −0.02 log-odds per
year, no discrete band structure). Assert `fars_condlogit_sex_trend_slope_frontal_{variant}`
recovers the planted per-decade slope within CI, and that its sign/rough magnitude agrees with the
implied slope across the discrete per-band estimates from the same synthetic data.

**7.7 Per-band-independent-nuisance sensitivity — homogeneous case.**
With planted data where `β_seat`/age truly are constant across bands (as in 7.3), assert the
per-band-independent-nuisance fit agrees with the pooled-nuisance fit within CI — proves pooling
doesn't distort the answer when the pooling assumption actually holds.

**7.8 No cross-channel leakage (regression/sanity check).**
Assert that `fars_condlogit_*` mixed-pair coefficients are numerically identical whether or not any
same-sex-eligible pairs exist in the test database — proves the two channels are computed from
disjoint frames and same-sex data cannot silently perturb the mixed-pair estimates.

**7.9 Method-consistency check.**
On a moderate synthetic dataset with a fixed seed, assert `cov_type="cluster"` analytic SEs and the
crash-clustered percentile bootstrap CI agree within a stated tolerance.

**7.10 Edge cases.**
Empty band cells, empty same-sex cohorts, and bands/cohorts with only concordant pairs all return
`None` for the affected estimand rather than raising (mirrors the existing
`test_empty_strata_do_not_crash` pattern).

**7.11 Power floor enforcement.**
Unit test that any estimand row with `n_discordant_pairs < 30` is written to `results` (the ledger
stays complete) but is excluded from the dashboard's headline treatment — assert the
dashboard-facing filtering function actually enforces the floor, not just that it's documented.

## 8. Dashboard/README changes

**Dashboard (`dashboard/app.py`):**
- New cached loader mirroring `load_bands`'s shape: age-adjusted per-band female OR (default
  pooled-nuisance model) alongside the existing naive unadjusted per-band chart, so the
  before/after of age-adjustment is visible in one view — this is the direct answer to the trend
  question and should be the most prominent new tile.
- Per-band points below the 30-discordant-pair floor are rendered visibly greyed out / flagged
  low-power, never silently hidden and never shown with the same visual weight as adequately
  powered points.
- New section for the same-sex seat baseline (`fars_samesex_seat_effect_male_*`,
  `fars_samesex_seat_effect_female_*`) shown side by side, with the interaction ratio
  (`fars_samesex_seatsex_interaction_*`) and its CI directly beneath — following the repo's
  existing "never show the confounded number alone" convention, extended here to "never show the
  interaction ratio without its same-sex baselines."
- Caveat block gains, in addition to v0's existing text (amended 2026-09-01: the full caveat
  text lives in an always-present expander in the page body, one click away, never in a
  tooltip; the identifying-assumption, null-result and power-floor points additionally appear
  in visible text or chart captions next to their numbers, so the "next to the number" rules
  in §3 and §6 still hold above the fold):
  - the epistemic caveat that this model is not verified against any specific published equation
    form (§2);
  - the pooled-nuisance-across-bands assumption and a link to where the per-band-independent
    sensitivity numbers live;
  - the same-sex severity-comparability caveat (§3, §6);
  - the quadratic-age functional-form caveat, noting the piecewise-linear sensitivity check exists
    and whether it agreed;
  - the 30-discordant-pair power floor rule, stated as a rule, not just implied by greyed-out
    points.
- Raw estimand table (existing `st.expander`) grows the new rows; separately expose
  `n_pairs` vs `n_discordant_pairs` as distinct columns there.

**README** (or equivalent top-level docs): document the new `statsmodels` runtime dependency, the
new estimand-name prefixes (`fars_condlogit_*`, `fars_samesex_*`) and what distinguishes each from
the `fars_doublepair_*` family, and the power-floor rule so anyone querying `results` directly
knows which rows are headline-safe.

## 9. Implementation plan mapped to repo files

- **`crashgap/analysis.py`** (extended, same module — cohort logic already lives here):
  - `occupant_frame(conn, years, frontal)` — occupant-grain SQL mirroring the existing pair-CTE,
    adding `age`, `mod_year`, banded `mod_year_band`, and seat position per occupant; row grain is
    per-occupant, still restricted to vehicles with exactly one eligible female and one eligible
    male.
  - `condlogit_frame(occupant_frame, band_mode)` — builds `y, x_seat, x_age, x_age_curv`, plus
    either band-dummy columns or `{x_const, x_trend}`, per discordant mixed-sex pair;
    `band_mode ∈ {"bands", "trend", "bands_separate"}` selects which female-effect structure to
    build, keeping the three models from ever being fit jointly.
  - `age_diff_quadratic(occ_f, occ_m)` / `age_diff_piecewise(occ_f, occ_m, knot=65)` — the two age
    functional-form builders, swappable behind one call site.
  - `fit_condlogit(frame)` → dataclass `CondLogit`, mirroring `DoublePair`/`Decomposition`'s shape
    (point/ci per coefficient, `n_pairs`, `n_discordant_pairs`, `n_crashes`).
  - `condlogit_by_band`, `condlogit_by_band_separate`, `condlogit_trend`, `condlogit_pooled` — thin
    wrappers over `condlogit_frame`/`fit_condlogit` selecting `band_mode`.
  - `samesex_pairs_frame(conn, years, frontal, sex)` — same occupant-eligibility predicate,
    `SUM(is_female) IN (0, 2)`, aggregated on seat instead of sex, discordant only.
  - `samesex_seat_effect(frame)`, `seatsex_interaction(male_frame, female_frame)` — reuse the
    existing `_bootstrap_crashes`/`_per_crash` resampling helpers.
  - `write_result` reused unchanged. A new `analyze_v1(conn, years, reps)` (or an extension of the
    existing `analyze()` behind a flag) writes all new estimands under one `run_ts`, leaving v0's 9
    rows untouched.
- **`pyproject.toml`**: add `statsmodels` to runtime dependencies (currently missing despite the
  bootstrap/CI work already implying `numpy` is pulled in transitively — flagged, worth confirming
  during implementation).
- **`tests/conftest.py`**: extend `add_pairs` with an `age` override per occupant (currently
  absent); add `add_samesex_pairs`, mirroring `add_pairs`'s shape but keyed on same-sex cohorts.
- **`tests/test_condlogit.py`** (new file): implements §7.1–§7.11.
- **`dashboard/app.py`**: new cached loaders (`load_condlogit_bands`, `load_condlogit_trend`,
  `load_samesex_interaction`), new page sections and caveat text per §8. No existing loader is
  removed; v0's tiles stay as-is.
- **README** (or project docs entry point): dependency and estimand-naming documentation per §8.

## 10. Explicit limitations for the tile's caveat block

- This model has not been checked against the exact published regression form for the model-year-
  band trend literature this project benchmarks against; it is a standard matched-pairs
  differenced-logit, not a verified reproduction.
- The default trend answer assumes `β_seat` and the age curve are constant across model-year bands;
  the per-band-independent-nuisance sensitivity numbers should be read alongside the pooled default
  before trusting a specific per-band point.
- Age enters as a closed-form quadratic on the within-pair difference. This will not capture a
  true age-mortality threshold effect (e.g., elevated risk past ~65–75); a piecewise-linear
  sensitivity variant is run, but neither form is a verified match to the true fatality-age curve.
- The seat × sex interaction is identified entirely from the same-sex-pair channel, which is a
  small fraction of the eligible cohort (two-occupant fatal crashes skew mixed-sex) and rests on a
  weaker identifying assumption than the mixed-pair design — same-sex pairs are not sex-paired
  within the same crash, so who rides with whom could correlate with crash severity or vehicle type
  in ways the mixed-pair design structurally rules out. A null interaction result is not strong
  evidence of no effect at this sample size.
- No covariate here reaches crash severity directly. If severity composition itself shifted across
  model-year bands (e.g. newer, safer vehicles overrepresented among *survivable* fatal crashes),
  that channel survives age-adjustment untouched and could still explain part of the naive trend.
- Any point estimate backed by fewer than 30 discordant pairs is not headlined; read it as a
  direction, not a number.

## 11. Rejected alternatives

- **Fully separate per-band fits as the primary/default trend estimator** (rather than pooled-
  nuisance-by-default): more power-hungry and risks small-sample/separation instability in the
  thinner older bands; kept only as the sensitivity-check role alongside the pooled default, not
  promoted to primary.
- **Recovering the seat × sex interaction from mixed pairs via a record-order occupant-labeling
  trick** (assign occupant "A"/"B" by a person-order field, add a `female × seat` term to one
  pooled mixed-pair regression): rejected — FARS conventionally numbers the driver first, so that
  labeling field is not independent of actual seat position, and the "interaction" collapses toward
  collinearity with the seat main effect, reproducing exactly the saturated 2-unknown system that
  makes the interaction unidentified from mixed pairs in the first place. The apparent one-
  regression convenience doesn't survive this check.
- **Fitting discrete band dummies and a continuous model-year trend term jointly in one equation**:
  rejected — band is a coarsened function of year, so the two are highly collinear together,
  risking a rank-deficient or unstable fit on exactly the trend question that's the publish-blocker.
  Kept as two separate models instead, cross-checked against each other.
- **A true N-per-stratum conditional-logit solver** (for a future cohort relaxing "exactly one
  eligible female, one eligible male" per vehicle to 3+ occupants): out of scope for v1 — the
  closed-form differenced-logit reduction is exact only for strata of size 2, which is the entire
  current cohort definition; flagged for future work if the cohort ever grows a third eligible
  occupant per vehicle.
- **Mantel–Haenszel per-band stratified odds ratios** as the primary trend estimator instead of the
  logit: not adopted — `statsmodels.Logit` on differenced covariates already reproduces exact
  conditional MLE for 2-occupant strata and integrates directly with the existing bootstrap/
  `write_result` machinery; could serve as an additional cross-check later, not required for v1.
- **Full 1975+ FARS ingestion for an exact published-trend replication**: rejected for v1 scope —
  large lift (early-schema drift across decades, unverified column availability across years) with
  limited incremental value, since the 2015–2024 window already targets the newest published band.
  Ingesting the intervening 2000–2014 years (to thicken the two oldest model-year bands, since those
  vehicles were common on the road then but are nearly extinct by 2015) is recommended as a scoped,
  valuable follow-up, not a v1 requirement.

## 12. Post-review amendments (adopted during implementation review)

The implementation review found two gaps in this document; both are corrected in the shipped code
and reflected above where table cells were wrong.

- **The same-sex channel needs an age adjustment this document did not specify.** On real
  2015–2024 data the male-male and female-female cohorts differ sharply in within-pair age
  structure (female-female pairs carry a right-front passenger 11+ years older than the driver in
  ~39% of discordant pairs; male-male passengers average slightly younger), and that composition
  alone manufactures most of the raw interaction log-ratio. Adopted fix: `_ageadj` variants of all
  three same-sex estimands (`fars_samesex_seat_effect_{male,female}_ageadj_frontal_{variant}`,
  `fars_samesex_seatsex_interaction_ageadj_frontal_{variant}`), refitting each baseline as a
  differenced logit with the section-2 quadratic age terms (passenger minus driver) and taking the
  interaction as the difference of the two `x_const` coefficients with a combined-SE normal CI.
  The `_ageadj` rows are the headline versions; the raw rows remain as diagnostics of the age
  correction's size. A planted test (equal true seat effects, skewed age composition) proves the
  adjusted estimator removes the spurious interaction the raw one reports.
- **Section 4 originally described the same-sex baselines as "driver-vs-passenger" ORs while the
  estimator computes passenger-vs-driver.** The table above now states the implemented
  orientation; any prose consuming these rows must say which seat the numerator counts.
- **The trend deliverable landed on the third branch of section 5's honest-deliverable set.** The
  per-band seat effect is not constant across bands on this data (age-adjusted 0.89 oldest → 1.34
  newest, CIs disjoint), so the default pooled-nuisance model's identifying assumption fails and
  the per-band female pattern flips direction between the pooled and separate-nuisance fits. The
  claim shipped is "trend direction unidentified on this window", not "consistent with shrinking"
  and not "still rising".

## 13. The 2000-2014 ingestion follow-up (executed 2026-09-01)

Section 11 scoped "ingest the intervening 2000-2014 years" as the follow-up that could move the
trend question. Executed; this section records what was verified, what was added, and where the
claims moved. All numbers below are frontal core and live in `results` under the 2000-2024 runs.

**Code-drift verification before widening the span** (distributions inspected on the real
2000/2004/2008/2009/2010/2014/2015 National CSV files; notes now pinned in `codebook.py`):

- `SEAT_POS` 11/13, `PER_TYP` 1/2, `INJ_SEV` 4 vs 0-3, `SEX` 1/2, `BODY_TYP` light-vehicle set:
  stable across the whole 2000-2024 span (the 30-33 pickup codes the set already carried ARE the
  2000-2019 coding).
- `AGE`: unknown = 99 through 2008, 998/999 from 2009. Both schemes fall outside the 16-96
  window; the top end must not be widened without handling the pre-2009 99 explicitly.
- `REST_USE`: the 2010 recode moved "none used" 0 -> 7 and reshuffled helmet codes; belted
  {1, 2, 3} keeps its meaning throughout, and "used, type unknown" (8) is excluded in both eras.
- `MAN_COLL` 2 ("Head-On" -> "Front-to-Front" in the 2010 recode): same code, continuous volumes
  across the 2009->2010 boundary (3,038 -> 2,851).
- `IMPACT1`: 2010+ adds side-specific codes 61-63/81-83 that drain clock points 2/3/9/10. The
  core zone {11, 12, 1} is continuous across the boundary; the WIDE variant's 10 and 2 select a
  narrower crash set after 2010, so cross-era comparisons of the wide variant carry an asterisk
  (documented in the codebook and on the dashboard).
- `MOD_YEAR` is 4-digit with 9999 sentinels in all these years; the 1970-2026 window already
  handles it.

**Two additions to the estimand contract** (both per frontal variant, both in the
`test_analyze_v1_writes_the_full_estimand_contract` contract):

- `fars_condlogit_sex_effect_frontal_{v}_band_{label}_vehage12` — the default band model refit on
  vehicles at most `VEHICLE_AGE_MAX` = 12 years old at crash. Rationale: with 25 calendar years
  pooled, an old band is observed at high vehicle ages in ways the newest band cannot be, and
  vehicle age is constant within a pair, so the within-vehicle design cannot difference it away;
  the cap puts bands on comparable vehicle-age support at the price of making band and calendar
  period nearly synonymous (model year = period - vehicle age; the three are linearly dependent
  and the design separates none of them).
- `fars_samesex_seatsex_interaction_ageadj_agecomparable_frontal_{v}` — the age-adjusted
  interaction refit on pairs with within-pair age gap <= `AGE_COMPARABLE_GAP` = 10 years, where
  the quadratic adjustment interpolates rather than extrapolates. This is the fragility check
  from section 12's review finding, promoted from a one-off diagnostic to a ledger row.

**Where the claims moved** (v1's "trend unidentified" and "interaction not robust" both
superseded — the dashboard and README carry the new framing):

- The 1970-1999 band went from 553 to 5,532 discordant pairs. The pre-2000 -> 2000s drop (1.26
  [1.19-1.33] -> 1.09 [1.03-1.15] default; 1.28 -> 1.11 separate-nuisance; 1.23 -> 1.05
  vehage12) has disjoint CIs under every specification: the direction of the published decline
  is reproduced in-window for the first time.
- No post-2000 pairwise band contrast has disjoint CIs under any specification. The 2010-2016
  band sits above the 2000s level in every specification; the newest band does in all but the
  separate-nuisance refit, where the two are indistinguishable (1.0977 vs 1.1057 - "at or above
  in every specification" is NOT a claim this data supports, per the 2026-09-01 review). A Wald
  contrast on the default pooled-nuisance fit reads 2017-2026 significantly ABOVE 2000-2009
  (diff +0.143, se 0.065, p ~ 0.03) - disclosed as a spec-dependent hint of a reversal, not a
  finding, because it does not survive the separate-nuisance refit. The newest band still flips
  with the nuisance choice (1.26 pooled vs 1.10 separate; the per-band seat effect stays the
  reason pooling is rejected: ~1.0 in older bands vs 1.34 [1.17-1.53] in 2017+). Continuous
  slope -0.025/decade [-0.062, +0.013]. The published continued shrink to ~5.8% is NOT
  reproduced; candidate explanations (2020-2024 period effects beyond the published study's
  2019 endpoint vs a genuine plateau) are stated as unresolvable in this design.
- The age-adjusted seat x sex interaction on the full window is +0.209 [0.121, 0.297] with
  n=7,140/4,744 discordant pairs, era-stable (2000-2009: +0.22, 2010-2017: +0.15, 2018-2024:
  +0.25), while the age-comparable refit reads +0.092 [-0.013, +0.198] — the honest range is the
  pair of numbers.
- v0's seat-balanced sex effect on the full window is 1.054 [1.018, 1.090] (significant, unlike
  the modern window's 1.041) — era-pooled, so it mostly reflects the older-vehicle penalty; the
  dashboard keeps the modern window as the headline for exactly this reason, via
  `modern_span()` and per-(family, window) latest-run selection.

**Runtime contract change:** `analyze-v1` now writes 84 rows per run (was 69), and the canonical
refresh is four runs — `analyze` + `analyze-v1` for the modern window and for the full window
(`deploy/refresh.sh`).
