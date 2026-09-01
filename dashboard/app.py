"""CrashGap dashboard.

Leads with the seat/sex decomposition, not the pooled ratio. The pooled
double-pair number is seat-confounded - seat position is the one attribute
that differs within every pair and it is ~73/27 collinear with sex - so
showing it alone would be the exact number a skeptic dismantles in one query.
It is shown here only as the thing being decomposed.

Every caveat is printed on the page, not tucked into a tooltip.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from crashgap.analysis import POWER_FLOOR

DB_PATH = os.environ.get("CRASHGAP_DB", "data/crashgap.db")

SEX = "fars_doublepair_sex_effect_frontal_"
SEAT = "fars_doublepair_seat_effect_rightfront_frontal_"
POOLED = "fars_doublepair_pooled_f_vs_m_seatconfounded_frontal_"
CONDLOGIT_SEX_BAND = "fars_condlogit_sex_effect_frontal_"
CONDLOGIT_TREND = "fars_condlogit_sex_trend_slope_frontal_"
SAMESEX_SEAT = "fars_samesex_seat_effect_"
SAMESEX_INTERACTION = "fars_samesex_seatsex_interaction_frontal_"

st.set_page_config(page_title="CrashGap", page_icon="🚗", layout="centered")


@st.cache_data(ttl=3600)
def load_results(db_path: str) -> pd.DataFrame:
    """v0 (`fars_doublepair_*`) and v1 (`fars_condlogit_*`/`fars_samesex_*`)
    rows are written by separate analyze() / analyze_v1() calls under their
    own run_ts, so a single global MAX(run_ts) would only ever surface
    whichever family ran last. Take the latest run of each family instead -
    the raw table and every downstream lookup below still gets one row per
    estimand, just not necessarily from the same instant."""
    if not Path(db_path).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        v0 = pd.read_sql_query(
            "SELECT * FROM results WHERE estimand LIKE 'fars_doublepair%' "
            "AND run_ts = (SELECT MAX(run_ts) FROM results "
            "WHERE estimand LIKE 'fars_doublepair%')", conn)
        v1 = pd.read_sql_query(
            "SELECT * FROM results WHERE estimand NOT LIKE 'fars_doublepair%' "
            "AND run_ts = (SELECT MAX(run_ts) FROM results "
            "WHERE estimand NOT LIKE 'fars_doublepair%')", conn)
        return pd.concat([v0, v1], ignore_index=True)
    finally:
        conn.close()


@st.cache_data(ttl=3600)
def load_strata(db_path: str) -> dict:
    from crashgap.analysis import decompose
    conn = sqlite3.connect(db_path)
    try:
        years = pd.read_sql_query(
            "SELECT MIN(year) y0, MAX(year) y1 FROM person WHERE source='fars'", conn)
        span = (int(years.y0[0]), int(years.y1[0]))
        return {v: decompose(conn, span, v, reps=500) for v in ("core", "wide", "headon")}
    finally:
        conn.close()


@st.cache_data(ttl=3600)
def load_bands(db_path: str) -> pd.DataFrame:
    from crashgap.analysis import by_model_year_band
    conn = sqlite3.connect(db_path)
    try:
        years = pd.read_sql_query(
            "SELECT MIN(year) y0, MAX(year) y1 FROM person WHERE source='fars'", conn)
        return by_model_year_band(conn, (int(years.y0[0]), int(years.y1[0])), reps=500)
    finally:
        conn.close()


def _discordant_count(cohort_def: str) -> int | None:
    n = json.loads(cohort_def).get("n_discordant_pairs")
    return int(n) if n is not None else None


@st.cache_data(ttl=3600)
def load_condlogit_bands(db_path: str, frontal: str = "core") -> pd.DataFrame:
    """Naive (unadjusted, seat-confounded) vs age-adjusted per-band female OR,
    side by side - the direct answer to the trend question (design doc
    section 5). Age-adjusted numbers come from the default pooled-nuisance
    band model (`fars_condlogit_sex_effect_frontal_{v}_band_{lo}-{hi}`);
    the `_agepiecewise`/`_separatenuisance` sensitivity variants are excluded
    here and read from the raw table instead."""
    naive = load_bands(db_path)
    results = load_results(db_path)
    prefix = f"{CONDLOGIT_SEX_BAND}{frontal}_band_"
    adj = results[
        results.estimand.str.startswith(prefix)
        & ~results.estimand.str.endswith(("_agepiecewise", "_separatenuisance"))
    ].copy()
    if adj.empty:
        naive["adj_point"] = None
        naive["adj_ci_lo"] = None
        naive["adj_ci_hi"] = None
        naive["n_discordant_pairs"] = None
        naive["headline_safe"] = True
        return naive
    adj["mod_year_band"] = adj.estimand.str.slice(len(prefix))
    adj["n_discordant_pairs"] = adj.cohort_def.apply(_discordant_count)
    adj = adj.rename(columns={"point": "adj_point", "ci_lo": "adj_ci_lo", "ci_hi": "adj_ci_hi"})
    merged = naive.merge(
        adj[["mod_year_band", "adj_point", "adj_ci_lo", "adj_ci_hi", "n_discordant_pairs"]],
        on="mod_year_band", how="left")
    merged["headline_safe"] = merged.n_discordant_pairs.isna() | (
        merged.n_discordant_pairs >= POWER_FLOOR)
    return merged


@st.cache_data(ttl=3600)
def load_condlogit_trend(db_path: str, frontal: str = "core") -> pd.Series | None:
    """Continuous model-year trend slope (log-odds per decade), the
    robustness cross-check against the discrete per-band pattern above."""
    results = load_results(db_path)
    hits = results[results.estimand == f"{CONDLOGIT_TREND}{frontal}"]
    return hits.iloc[0] if not hits.empty else None


@st.cache_data(ttl=3600)
def load_samesex_interaction(db_path: str, frontal: str = "core") -> dict:
    """Same-sex passenger-vs-driver seat baselines (male, female), raw and
    age-adjusted, plus the two interaction log-ratios - never shown without
    the baselines, per the repo's existing 'never show the confounded number
    alone' convention (design doc section 8). The age-adjusted rows are the
    headline versions: the two cohorts differ sharply in within-pair age
    structure, and that composition loads directly onto the raw ORs."""
    results = load_results(db_path)
    row = results.set_index("estimand")
    out = {}
    for sex in ("male", "female"):
        name = f"{SAMESEX_SEAT}{sex}_frontal_{frontal}"
        out[sex] = row.loc[name] if name in row.index else None
        name = f"{SAMESEX_SEAT}{sex}_ageadj_frontal_{frontal}"
        out[f"{sex}_ageadj"] = row.loc[name] if name in row.index else None
    name = f"{SAMESEX_INTERACTION}{frontal}"
    out["interaction"] = row.loc[name] if name in row.index else None
    name = f"fars_samesex_seatsex_interaction_ageadj_frontal_{frontal}"
    out["interaction_ageadj"] = row.loc[name] if name in row.index else None
    return out


def piecewise_agreement_text(condlogit_bands: pd.DataFrame) -> str:
    """Compares the quadratic default per-band female OR against the
    `_agepiecewise` sensitivity variant and states, in one sentence, whether
    they agree — the dynamic half of the age-functional-form caveat rather
    than a static claim that could go stale."""
    results = load_results(DB_PATH)
    prefix = f"{CONDLOGIT_SEX_BAND}core_band_"
    pw = results[results.estimand.str.startswith(prefix)
                & results.estimand.str.endswith("_agepiecewise")].copy()
    if pw.empty or condlogit_bands.empty:
        return "the piecewise-linear check could not be compared in this run (no rows available)."
    pw["mod_year_band"] = pw.estimand.str.slice(len(prefix)).str.removesuffix("_agepiecewise")
    merged = condlogit_bands.merge(pw[["mod_year_band", "point"]], on="mod_year_band", how="inner")
    merged = merged.dropna(subset=["adj_point", "point"])
    if merged.empty:
        return "the piecewise-linear check could not be compared in this run (no overlapping bands)."
    max_diff = float(((merged["point"] - merged["adj_point"]).abs() / merged["adj_point"]).max())
    if max_diff < 0.02:
        return (f"it agrees closely with the quadratic default (largest per-band point-estimate "
                f"difference: {max_diff * 100:.1f}%).")
    return (f"it disagrees materially with the quadratic default in at least one band (largest "
            f"per-band point-estimate difference: {max_diff * 100:.1f}%) — treat the trend "
            f"conclusion as an open risk, not settled.")


def pct(x: float | None) -> str:
    return f"{(x - 1) * 100:+.1f}%" if x else "n/a"


st.title("CrashGap")
st.caption("Female-vs-male fatality risk in the same crash, computed live from "
           "public-domain NHTSA FARS data — with the seat-position confound "
           "separated out.")

results = load_results(DB_PATH)
if results.empty:
    st.warning("No results yet. Run `crashgap ingest` and `crashgap analyze` first.")
    st.stop()

row = results.set_index("estimand")
sex, seat, pooled = row.loc[SEX + "core"], row.loc[SEAT + "core"], row.loc[POOLED + "core"]
strata = load_strata(DB_PATH)

left, right = st.columns(2)
left.metric("Being female (seat-balanced)", pct(sex.point),
            delta=f"95% CI {pct(sex.ci_lo)} to {pct(sex.ci_hi)}", delta_color="off")
right.metric("Sitting in the right-front seat", pct(seat.point),
             delta=f"95% CI {pct(seat.ci_lo)} to {pct(seat.ci_hi)}", delta_color="off")

st.markdown(f"""
Among belted front-seat mixed-sex pairs riding in the **same vehicle** in a fatal frontal crash
(FARS {sex.fars_years}, {int(sex.n_pairs):,} pairs in {int(sex.n_crashes):,} crashes), the
**seat matters more than the sex**. Sitting right-front rather than driving raises the relative
fatality risk by **{pct(seat.point)}** and that is statistically significant. Being female raises
it by **{pct(sex.point)}**, and in this 10-year window that is **not** distinguishable from zero.
""")

st.error(f"""**Why the obvious number is wrong.** Pool the pairs without splitting by seat and you
get a female-vs-male ratio of **{pooled.point:.2f}×** — which reads like a {pct(pooled.point)}
female penalty and is the number this project nearly shipped. It is mostly the seat effect in
disguise: women are the passenger in {strata['core'].n_pairs_she_passenger:,} of the
{strata['core'].n_pairs:,} pairs and the driver in only
{strata['core'].n_pairs_she_drives:,}, so "female" and "right-front" are ~73/27 collinear. Split
them and the female share of that gap largely evaporates.""")

st.subheader("The split, in raw counts")
core = strata["core"]
st.dataframe(pd.DataFrame([
    {"configuration": "she is the passenger, he drives",
     "pairs": core.n_pairs_she_passenger, "ratio (she died : he died)": core.rr_she_passenger},
    {"configuration": "she drives, he is the passenger",
     "pairs": core.n_pairs_she_drives, "ratio (she died : he died)": core.rr_she_drives},
]), hide_index=True)
st.caption("If sex alone drove the outcome these two ratios would agree. They don't: the ratio "
           "flips below 1.0 when she is the one driving, which is the seat effect showing "
           "through. The geometric mean of the two cancels seat and isolates sex; their "
           "geometric ratio cancels sex and isolates seat (Evans' double-pair decomposition).")

st.subheader("Does the frontal definition change the story?")
sens = pd.DataFrame([
    {"frontal cut": v,
     "sex effect": pct(d.sex_effect),
     "sex 95% CI": f"{pct(d.sex_ci[0])} to {pct(d.sex_ci[1])}",
     "sex sig.": "yes" if d.sex_is_significant else "no",
     "seat effect": pct(d.seat_effect),
     "seat sig.": "yes" if d.seat_is_significant else "no",
     "pairs": d.n_pairs}
    for v, d in strata.items()
])
st.dataframe(sens, hide_index=True)
st.warning("""**Read this row-by-row, because the cuts disagree and that is informative.** In the
two broad frontal cuts the female effect is small and not significant. In the strict head-on cut
(`MAN_COLL=2`) it is **+11.5% and significant** — and the seat effect *reverses*, with the driver
faring worse. That reversal is physically plausible rather than noise: head-on collisions are
predominantly offset toward the driver's side, which is why the driver-side small-overlap crash
test existed for six years before a passenger-side version was added. The honest reading is that
the female effect is somewhere in the low single digits to low double digits depending on crash
geometry, and this window cannot pin it down further. Anyone quoting only the head-on row is
cherry-picking; so is anyone quoting only the core row.""")

st.subheader("By vehicle model year (no trend claim)")
st.dataframe(load_bands(DB_PATH).rename(columns={
    "mod_year_band": "vehicle model years", "f_only": "she died, he survived",
    "m_only": "he died, she survived", "n_pairs": "mixed-sex pairs", "rr": "pooled RR",
    "ci_lo": "CI low", "ci_hi": "CI high",
}), hide_index=True)
st.caption("These bands use the pooled (seat-confounded) ratio, their CIs all overlap, and they "
           "are unadjusted for occupant age — they cannot confirm or refute the shrinking trend "
           "that Atwood, Noh & Craig (2023) find over 45 years of FARS. Don't quote a trend off "
           "them.")

st.subheader("The trend question, age-adjusted (v1)")
st.markdown("""
The naive bands above are pooled (seat-confounded) and unadjusted for occupant age. This tile is
the direct test of whether age adjustment reconciles that with the published shrinking trend: a
within-vehicle differenced conditional logit that holds seat position and occupant age constant
across bands (`x_seat`, `x_age`, `x_age_curv` shared, one female coefficient per model-year band)
— see Method below for the naive/adjusted numbers side by side.
""")
condlogit_bands = load_condlogit_bands(DB_PATH)


def _grey_low_power(row: pd.Series) -> list[str]:
    if row["headline safe"] == "low power":
        return ["color: #888888; background-color: rgba(128,128,128,0.12)"] * len(row)
    return [""] * len(row)


band_display = condlogit_bands.rename(columns={
    "mod_year_band": "vehicle model years", "rr": "naive RR (unadjusted, pooled)",
    "ci_lo": "naive CI low", "ci_hi": "naive CI high",
    "adj_point": "age-adjusted female OR", "adj_ci_lo": "adjusted CI low",
    "adj_ci_hi": "adjusted CI high", "n_discordant_pairs": "discordant pairs",
})

# Reuse the headline_safe column load_condlogit_bands already computed from
# POWER_FLOOR, rather than re-deriving the same rule here with a second
# literal threshold that could drift out of sync with it.
band_display["headline safe"] = band_display["headline_safe"].map(
    lambda ok: "ok" if ok else "low power")
cols = ["vehicle model years", "naive RR (unadjusted, pooled)", "naive CI low", "naive CI high",
        "age-adjusted female OR", "adjusted CI low", "adjusted CI high", "discordant pairs",
        "headline safe"]
st.dataframe(band_display[cols].style.apply(_grey_low_power, axis=1), hide_index=True)
st.caption(f"Rows flagged **low power** carry fewer than the {POWER_FLOOR}-discordant-pair floor "
           "this project requires before headlining a point estimate (design doc section 5) — "
           "greyed out here, never dropped from the table and never shown at the same visual "
           "weight as an adequately powered band. Read a low-power row as a direction, not a "
           "number.")

trend = load_condlogit_trend(DB_PATH)
if trend is not None and pd.notna(trend.point):
    st.caption(f"Continuous-trend cross-check (fit separately from the band dummies, never "
               f"jointly — band is a coarsened function of year): **{trend.point:+.4f}** log-odds "
               f"per decade (95% CI {trend.ci_lo:+.4f} to {trend.ci_hi:+.4f}). Its sign should "
               f"agree with the discrete per-band pattern above; it is a robustness check, not a "
               f"third headline number.")

st.subheader("Same-sex seat baseline and the seat × sex interaction (v1)")
st.markdown("""
A `female × seat` term cannot be recovered from mixed-sex pairs at all — within any discordant
mixed-sex pair her indicator minus his is identically 1, so that term is algebraically the seat
main effect, not an interaction (design doc section 6). This channel routes around that: the
right-front-passenger-vs-driver fatality odds computed separately within male-male and
female-female discordant pairs, shown side by side so the interaction ratio beneath can be read
against a sane baseline — never on its own.

**The raw baselines are age-confounded and are shown only as diagnostics.** The two cohorts differ
sharply in *who sits where at what age*: female-female pairs carry a much older right-front
passenger far more often than male-male pairs do, and age — not seat, not sex — is the strongest
single predictor of who dies. The headline numbers here are therefore the **age-adjusted** fits
(quadratic within-pair age difference, same machinery as the mixed-sex model); the raw ORs sit
beneath them so the size of the age correction is visible rather than hidden.
""")
samesex = load_samesex_interaction(DB_PATH)
male_seat, female_seat = samesex["male"], samesex["female"]
male_adj, female_adj = samesex["male_ageadj"], samesex["female_ageadj"]
interaction = samesex["interaction"]
interaction_adj = samesex["interaction_ageadj"]


def _row_headline_safe(row: pd.Series | None) -> bool:
    """A results row counts as renderable only if it exists, its point isn't
    the NaN a NULL SQL REAL round-trips to (a plain `is not None` check does
    not catch this - `row` itself is a real Series, not None, even when
    every value in it is NaN), and it clears the same power floor the band
    table above enforces. This is the most power-starved section on the
    page (design doc section 6), so it is exactly where a missing/silent
    floor check would first bite."""
    if row is None or pd.isna(row.point):
        return False
    n = _discordant_count(row.cohort_def)
    return n is None or n >= POWER_FLOOR


male_ok, female_ok = _row_headline_safe(male_adj), _row_headline_safe(female_adj)
interaction_ok = _row_headline_safe(interaction_adj)
raw_ok = all(_row_headline_safe(r) for r in (male_seat, female_seat, interaction))

sx_left, sx_right = st.columns(2)
if male_ok:
    sx_left.metric("Male-male pairs: right-front vs. driving (age-adjusted)",
                   f"{male_adj.point:.3f}×",
                   delta=f"95% CI {male_adj.ci_lo:.3f} to {male_adj.ci_hi:.3f}",
                   delta_color="off")
else:
    sx_left.info("Male-male seat baseline not available or below the power floor in this run.")
if female_ok:
    sx_right.metric("Female-female pairs: right-front vs. driving (age-adjusted)",
                    f"{female_adj.point:.3f}×",
                    delta=f"95% CI {female_adj.ci_lo:.3f} to {female_adj.ci_hi:.3f}",
                    delta_color="off")
else:
    sx_right.info("Female-female seat baseline not available or below the power floor in this run.")

if male_ok and female_ok and interaction_ok:
    st.metric("Age-adjusted interaction: log(female seat effect / male seat effect)",
              f"{interaction_adj.point:+.3f}",
              delta=f"95% CI {interaction_adj.ci_lo:+.3f} to {interaction_adj.ci_hi:+.3f}",
              delta_color="off")
    if raw_ok:
        st.caption(f"Unadjusted diagnostics: male-male {male_seat.point:.3f}× "
                   f"[{male_seat.ci_lo:.3f}, {male_seat.ci_hi:.3f}], female-female "
                   f"{female_seat.point:.3f}× [{female_seat.ci_lo:.3f}, {female_seat.ci_hi:.3f}], "
                   f"raw log-ratio {interaction.point:+.3f} [{interaction.ci_lo:+.3f}, "
                   f"{interaction.ci_hi:+.3f}]. The gap between the raw and age-adjusted rows IS "
                   f"the age-composition confound — do not quote the raw asymmetry as a seat or "
                   f"sex effect.")
    st.caption("**Read the adjusted number against a weaker identifying assumption than every "
               "other number on this page**: same-sex pairs are not sex-paired within the same "
               "crash, so who rides with whom could correlate with crash severity or vehicle type "
               "in ways the mixed-pair design structurally rules out — and the age adjustment "
               "leans on the quadratic age model most heavily in exactly the large-age-gap pairs "
               "that dominate what remains of the effect. This channel is also the most "
               "power-starved on the page — two-occupant fatal crashes skew heavily mixed-sex — "
               "so a null result here is not strong evidence of no effect.")
else:
    st.info(f"Same-sex interaction estimand not available, or below the {POWER_FLOOR}-discordant-"
            f"pair power floor, in this run.")

st.subheader("Read this before quoting anything here")
st.markdown("""
- **These are within-crash relative risks given a fatal frontal crash — never population rates.**
  FARS records fatal crashes only; every crash here already killed someone.
- **Not severity-adjusted beyond what the shared vehicle controls.** Same vehicle, same impact,
  same crash — but occupant age still differs within pairs (women here average ~1.4 years younger).
  The v0 `fars_doublepair_*` rows do not adjust for that; the v1 `fars_condlogit_*` and `_ageadj`
  rows do (quadratic within-pair age difference).
- **The published severity-adjusted gap is a different, larger quantity.** Bose (2011) reports
  MAIS 3+ odds ~1.47 from crash-investigation data. That needs CISS/NASS-CDS with injury grading
  and survey weights — the next rung, not this tile. Do not read this page as refuting it.
- **Coded fields are coarse.** Belt use and injury severity are police-coded (KABCO), not measured.
""")

st.subheader("v1 model caveats — read these next to the age-adjusted and interaction numbers")
st.markdown("""
- **Not a verified reproduction of any published regression form.** This is a standard matched-pairs
  differenced-logit (the conditional-likelihood identity is textbook), built to answer the trend and
  interaction questions on this data — it has not been checked against the exact model-year-band
  equations the literature this project benchmarks against actually uses.
- **The default trend answer pools `β_seat` and the age curve across model-year bands.** That is an
  assumption, not a fact. The `_separatenuisance` rows in the raw table below refit those nuisance
  parameters independently inside each band; read them alongside the pooled default before trusting
  a specific per-band point.
- **Age enters as a closed-form quadratic on the within-pair age difference.** This will not capture
  a true age-mortality threshold (elevated risk past roughly 65–75). A piecewise-linear sensitivity
  variant (knot at 65, `_agepiecewise` rows below) is run alongside it — {piecewise_agreement}
  Neither functional form is a verified match to the true fatality-age curve.
- **The seat × sex interaction is identified entirely from the same-sex-pair channel above**, a small
  fraction of the eligible cohort (two-occupant fatal crashes skew mixed-sex) resting on a weaker
  identifying assumption than the mixed-pair design: same-sex pairs are not sex-paired within the
  same crash, so who rides with whom could correlate with crash severity or vehicle type. A null
  interaction result is not strong evidence of no effect at this sample size.
- **The raw same-sex baselines are age-confounded — quote only the `_ageadj` rows.** The male-male
  and female-female cohorts differ sharply in within-pair age structure (female-female pairs carry
  a much older right-front passenger far more often), so the unadjusted ORs mix the seat effect
  with *who sits where at what age*. The age-adjusted rows correct this within the quadratic age
  model's reach, but what remains of the interaction is carried disproportionately by
  large-age-gap pairs, where that model extrapolates furthest — treat the residual as fragile, not
  settled.
- **No covariate here reaches crash severity directly.** If severity composition itself shifted
  across model-year bands (e.g. newer, safer vehicles overrepresented among *survivable* fatal
  crashes), that channel survives age-adjustment untouched and could still explain part of the naive
  trend.
- **The 30-discordant-pair power floor is a rule, not a suggestion.** Any point estimate backed by
  fewer than 30 discordant pairs is written to the results ledger but is never headlined on this
  page — read it as a direction, not a number.
""".format(piecewise_agreement=piecewise_agreement_text(condlogit_bands)))

st.subheader("Method")
st.markdown("""
Evans double-pair design: only vehicles carrying **exactly one eligible male and one eligible
female** (both belted, front outboard, age 16–96, light vehicle) enter, which holds crash severity,
delta-V, vehicle and impact direction physically constant within the vehicle. The estimate is the
ratio of discordant outcomes. Running it separately in the two seat configurations and taking
geometric means splits it into a sex effect and a seat effect — with *f* the female multiplier and
*s* the right-front multiplier, the two configurations measure *f·s* and *f/s*, so `sqrt(product)`
recovers *f* and `sqrt(quotient)` recovers *s*.

Frontal = IMPACT1 ∈ {11, 12, 1} (Forman 2019); the wider fan and head-on cuts are shown above.
Confidence intervals are percentile bootstraps resampling **whole crashes**, because occupants of
one crash are not independent observations; both strata are resampled inside one replicate so the
two effects stay mutually consistent.

The underlying finding is settled science (Evans 1988; Bose 2011; Forman 2019; Atwood, Noh & Craig
2023). What CrashGap adds is the live, versioned, queryable form: every number here sits in a
`results` table stamped with run timestamp, git commit, FARS years and the full serialized cohort
definition.

**v1 adds a within-vehicle differenced conditional logit** for the age-adjusted and interaction
tiles above. For a 2-occupant stratum the conditional MLE has a closed form: the stratum likelihood
depends only on the difference of the linear predictor between the two occupants, so fitting
reduces to a logistic regression of `y = 1{female occupant died}` on covariates defined as her
value minus his (`x_seat`, and an age difference — quadratic by default) over discordant pairs
only, with cluster-robust standard errors on `(year, st_case)`. It nests v0's decomposition above as
the special case with no age variation. The seat × sex interaction is not fit from these mixed
pairs at all (see the caveats above) — it comes from the separate same-sex channel.
""")

with st.expander("Cohort definition (serialized, as stored with the result)"):
    st.json(json.loads(sex.cohort_def))

with st.expander("All estimands from this run"):
    table = results.copy()
    table["n_discordant_pairs"] = table.cohort_def.apply(_discordant_count)
    st.dataframe(
        table[["estimand", "point", "ci_lo", "ci_hi", "n_pairs", "n_discordant_pairs",
              "n_crashes"]],
        hide_index=True)
    st.caption("`n_pairs` is the eligible cohort (incl. concordant pairs); `n_discordant_pairs` is "
               "the informative subset the fit actually used — blank for v0 rows, which don't "
               "serialize that split. A discordant-only fit is correct for the odds ratio, but a "
               "bare `n_pairs` on its own overstates effective power.")

st.caption(f"FARS {sex.fars_years} · run {sex.run_ts} · commit {sex.git_commit} · "
           "Data: NHTSA FARS (US public domain).")
