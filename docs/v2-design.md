# v2 design — the CISS / NASS-CDS severity rung

2026-09-01. The rung that reaches the severity-adjusted odds ratios the literature reports
(Bose 2011: MAIS 3+ OR 1.47 on NASS-CDS 1998-2008; Forman 2019: AIS 2+/3+ frontal; Craig 2024:
MAIS 2+ OR ~1.75, CISS-era). FARS knows who died; it has no injury grading, no delta-V and no
sampling weights, so v0/v1 could never touch these estimands. CISS and NASS-CDS are weighted
tow-away *samples* with hospital-grade AIS injury coding and reconstructed delta-V — a different
estimand family, kept in different tables and never pooled with the FARS channels.

## 1. Sources (verified live 2026-09-01, S3 listing on static.nhtsa.gov)

- **CISS 2017-2024** — `nhtsa/downloads/CISS/{year}/CISS_{year}_CSV_files.zip`. Tables used:
  `OCC.csv` (occupant: AGE, SEX, SEATLOC, ROLE, BELTUSE, HEIGHT, WEIGHT, BMI, MAIS, ISS,
  CASEWGT, PSU, PSUSTRAT, CASENUMBER, VEHNO, OCCNO), `GV.csv` (BODYTYPE, MODELYR, DVTOTAL,
  DVBASIS), `CDC.csv` (per-event damage: CDCPLANE, OCLOCK, DVRANK, EVENTNO — the frontal
  classifier). 2016 was a partial launch year with no data zip; excluded.
- **NASS-CDS 2000-2015** — per-year `accident/oa/gv/ve.sas7bdat` under `Formatted Data/` (2009+)
  or `PCSAS/` (2001-2008), with 2000 only inside `PCSAS/PCSAS.zip`. `oa` (AGE, SEX, SEATPOS,
  ROLE, MANUSE, HEIGHT, WEIGHT, MAIS, and 2010+ also MAIS08, RATWGT, PSU, CASENO), `accident`
  (PSUSTRAT), `gv` (BODYTYPE, MODELYR, DVTOTAL), `ve` (GAD1, DOF1 — the frontal classifier).
  1998-1999 are deliberately out: Bose's window reached back to 1998, but pre-2000 files add two
  more schema vintages for two years of data.

## 2. Code sets (verified against FORMAT22.sas value labels for CISS and empirically across
NASS 2000/2005/2010; pinned in `severity_codebook.py`)

- **Sex**: 1 = male; **female = {2, 3, 4, 5, 6}** (2 plus four pregnancy codes) in BOTH sources;
  9 / NaN unknown. Missing the pregnancy codes would silently drop pregnant women — the exact
  cohort the crash-test-dummy critique cares about.
- **Belted = {2, 3, 4}** (shoulder / lap / lap+shoulder) in both (CISS `BELTUSE`, NASS `MANUSE`);
  5 "type unknown" excluded, mirroring the FARS convention of excluding used-type-unknown.
- **Seat**: 11 driver / 13 right-front — same convention as FARS.
- **MAIS**: 0-6 valid; excluded codes differ by AIS revision: NASS `MAIS` (AIS90) uses 7 =
  injured-unknown; CISS `MAIS` (AIS2015) uses 9/99; NASS `MAIS08` uses 9. NaN = no injury
  workup.
- **Light vehicle**: CISS `BODYTYPE` and NASS `BODYTYPE` both use the FARS numeric family, so
  the existing `codebook.BODY_TYP_LIGHT_VEHICLE` set is reused verbatim — one pinned set across
  all three sources.
- **Frontal**: damage-based, symmetric with v0/v1's 11-1 o'clock core zone. NASS: `GAD1 = 'F'`
  and `DOF1 in {11, 12, 1}` (ve). CISS: on the vehicle's primary damage event (lowest `DVRANK`,
  ties and rank-unknowns by lowest `EVENTNO`): `CDCPLANE = 'F'` and `OCLOCK in {11, 12, 1}`.
  Note this is not identical to Bose's frontal definition (GAD F with a wider PDOF fan); the
  benchmark comparison says so.
- **Sentinels**: CISS AGE 999, HEIGHT 999, WEIGHT 999, BMI 99.9, DVTOTAL 999, MODELYR 9999 all
  → NULL at ingest; NASS uses SAS missing.

## 3. Storage

One denormalized analysis-grain table `sev_occupant` (occupant row carrying its vehicle's
fields), PK `(source, year, psu, case_id, veh_no, occ_no)`, alongside — never inside — the FARS
tables. The v0 plan reserved `person.mais/weight` for this rung; that reservation is superseded:
the severity sources are occupant-workup-grain with design metadata (PSU, stratum, weight,
delta-V, BMI, dual AIS coding) that has no FARS analogue, and wedging them into the FARS-shaped
`person`/`vehicle` tables would overload `(st_case)` join semantics for zero query benefit. The
unused nullable columns stay where they are; `db.py` says so.

## 4. Cohort

Front-outboard (11/13) adult (16-96, the v0/v1 window) belted occupants of light vehicles in
frontal tow-away crashes (both sources are tow-away by design), with known sex and known MAIS.
Complete-case on the model covariates; every filter and its survivor count is serialized into
`cohort_def`. Known selection: NASS occupant rows without completed injury workup (~40% of oa
rows, NaN MAIS/SEATPOS blocks) drop out — the same restriction every published NASS analysis
makes, stated rather than hidden.

## 5. Model and estimands

Design-weighted logistic pseudo-MLE (survey convention): outcome `MAIS >= k`,

- tier `base`: female + age + age² + right-front seat + model-year-band dummies
- tier `dv`: base + DVTOTAL + DVTOTAL², on the delta-V-known subset (~45-60% of vehicles;
  missingness is selective and the tiers exist to show what the subset does to the answer)
- tier `dv_anthro`: dv + height + BMI, on the subset where both are known — the "is it just
  body size" check, the crux of the crash-test-dummy debate (Bose adjusted for height/BMI)

Outcomes: `mais2plus` (Craig's quantity) and `mais3plus` (Bose's), per source. NASS extra:
`_ais08` sensitivity refits on `MAIS08` for the dual-coded 2010-2015 years, sizing the AIS
revision effect that sits between NASS-era and CISS-era benchmarks.

**Variance**: Taylor linearization for a stratified, with-replacement, single-stage-cluster
design — strata = `PSUSTRAT`, clusters = `PSU`, weights = `CASEWGT`/`RATWGT`:
`V = A⁻¹ B A⁻¹`, `A = Σ wᵢ pᵢ(1-pᵢ) xᵢxᵢᵀ`,
`B = Σ_h n_h/(n_h-1) Σ_j (z_hj - z̄_h)(z_hj - z̄_h)ᵀ`, `z_hj = Σ_{i∈PSU hj} wᵢ xᵢ (yᵢ - pᵢ)`,
CIs on a t distribution with **df = n_PSU − n_strata** (the svy convention; CISS: 32 − 12 = 20,
so intervals are meaningfully wider than normal-theory ones). A singleton-stratum PSU centers
against the grand mean rather than crashing. Tested against statsmodels' cluster sandwich in the
single-stratum case (they must agree up to the n_h/(n_h−1) factor and df) and against a planted
stratified-cluster simulation.

**Multi-year pooling**: years pooled within source; PSUs are persistent physical sites, so
same-PSU rows across years belong to ONE cluster and (PSUSTRAT, PSU) carry over unchanged.
Weight rescaling by 1/n_years is omitted — it changes neither logit coefficients nor linearized
SEs. NASS and CISS are never pooled with each other (different designs, eras, AIS revisions).

**Weight instability** (the Viano 2025 critique): every fit's `cohort_def` carries
`sum_weight`, `kish_neff` = (Σw)²/Σw², and `weight_max_over_median`; a `_wtrim95` sensitivity
refit caps weights at the cohort's 95th percentile. Power floor: `n_events` (unweighted
positive-outcome count) ≥ 30, the v2 analogue of the discordant-pair floor, enforced by the
same `headline_safe` machinery (it reads `n_events` when present).

## 6. What v2 does NOT claim

- No reproduction claim of Bose/Craig's exact numbers — cohorts, frontal definitions, AIS
  revisions and covariate sets all differ in documented ways; the benchmark table states the
  published number next to ours and the differences beside it.
- No pooling with FARS channels; no fatality estimand (FARS owns death; MAIS owns severity).
- No causal reading of the anthro tier: height/BMI are partly *mediators* of a sex effect, not
  confounders — adjusting them away answers "net of body size", not "is the gap real".

## 7. Results (run 2026-09-01, canonical numbers in `results` under the v2 run)

Ingested: CISS 2017-2024 = 52,943 occupant rows; NASS-CDS 2000-2015 = 150,897. Frontal
severity cohorts: CISS 13,145 (1,984 MAIS 2+ events, design df 28), NASS 28,220 (5,331 events,
design df 15). Headline cells (female OR, t-based CI):

| estimand | base | dv | dvanthro | wtrim95 | published |
|---|---|---|---|---|---|
| CISS MAIS 2+ | 1.54 [1.18-2.02] | 1.91 [1.23-2.96] | 2.01 [1.21-3.33] | 1.39 [1.09-1.77] | Craig 2024 ~1.75 |
| CISS MAIS 3+ | 1.27 [0.78-2.07] | 1.88 [0.89-3.96] | 1.27 [0.72-2.21] | 1.16 [0.78-1.74] | — |
| NASS MAIS 2+ | 1.72 [1.20-2.46] | 2.02 [1.40-2.90] | 2.63 [1.76-3.94] | 1.45 [1.28-1.65] | — |
| NASS MAIS 3+ | 1.21 [0.93-1.57] | 1.50 [1.10-2.04] | 1.51 [1.20-1.90] | 1.15 [0.92-1.43] | Bose 2011 1.47 |

Reading: (a) delta-V adjustment raises every estimate — women's frontal crashes carry lower
delta-V, so unadjusted tiers understate the conditional injury odds, the direction the
literature reports; (b) the dv tiers land on the published numbers (NASS MAIS 3+ dv 1.50 vs
Bose 1.47; CISS MAIS 2+ dv 1.91 vs Craig ~1.75) without being tuned to them; (c) body size
does not absorb the effect (only the underpowered CISS MAIS 3+ cell falls under anthro, n.s.
in both tiers); (d) both pre-registered fragilities are real and measured: max/median weights
run ~115x (CISS) and ~500x (NASS) with Kish effective n near 10-15% of nominal, and trimming
moves points by 0.1-0.3 without changing any significance verdict; the AIS revision moves NASS
MAIS 3+ from 1.21 (AIS90) to 0.93 (AIS2008) on the dual-coded subset — part of the
Bose-vs-Craig benchmark gap is injury coding, not crash physics; (e) NASS MAIS 3+ base (the
closest cohort to Bose's) is not significant on its own — only the delta-V-adjusted tier is,
which is also how Bose's model was specified.
