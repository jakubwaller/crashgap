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

DB_PATH = os.environ.get("CRASHGAP_DB", "data/crashgap.db")

SEX = "fars_doublepair_sex_effect_frontal_"
SEAT = "fars_doublepair_seat_effect_rightfront_frontal_"
POOLED = "fars_doublepair_pooled_f_vs_m_seatconfounded_frontal_"

st.set_page_config(page_title="CrashGap", page_icon="🚗", layout="centered")


@st.cache_data(ttl=3600)
def load_results(db_path: str) -> pd.DataFrame:
    if not Path(db_path).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(
            "SELECT * FROM results WHERE run_ts = (SELECT MAX(run_ts) FROM results)",
            conn)
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

st.subheader("Read this before quoting anything here")
st.markdown("""
- **These are within-crash relative risks given a fatal frontal crash — never population rates.**
  FARS records fatal crashes only; every crash here already killed someone.
- **Not severity-adjusted beyond what the shared vehicle controls.** Same vehicle, same impact,
  same crash — but occupant age still differs within pairs (women here average ~1.4 years younger),
  and that is not adjusted for in this version.
- **The published severity-adjusted gap is a different, larger quantity.** Bose (2011) reports
  MAIS 3+ odds ~1.47 from crash-investigation data. That needs CISS/NASS-CDS with injury grading
  and survey weights — the next rung, not this tile. Do not read this page as refuting it.
- **Coded fields are coarse.** Belt use and injury severity are police-coded (KABCO), not measured.
""")

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
""")

with st.expander("Cohort definition (serialized, as stored with the result)"):
    st.json(json.loads(sex.cohort_def))

with st.expander("All estimands from this run"):
    st.dataframe(results[["estimand", "point", "ci_lo", "ci_hi", "n_pairs", "n_crashes"]],
                 hide_index=True)

st.caption(f"FARS {sex.fars_years} · run {sex.run_ts} · commit {sex.git_commit} · "
           "Data: NHTSA FARS (US public domain).")
