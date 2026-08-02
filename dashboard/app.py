"""CrashGap dashboard: one honest tile.

Reads the latest run from the results table plus the model-year band check.
Every caveat is printed on the page, not tucked into a tooltip - the tile is
only defensible with its interpretation label attached.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = os.environ.get("CRASHGAP_DB", "data/crashgap.db")

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
def load_bands(db_path: str) -> pd.DataFrame:
    from crashgap.analysis import by_model_year_band
    conn = sqlite3.connect(db_path)
    try:
        years = pd.read_sql_query(
            "SELECT MIN(year) y0, MAX(year) y1 FROM person WHERE source='fars'", conn)
        return by_model_year_band(conn, (int(years.y0[0]), int(years.y1[0])))
    finally:
        conn.close()


st.title("CrashGap")
st.caption("The female-vs-male road-crash fatality gap, computed live from "
           "public-domain NHTSA FARS data.")

results = load_results(DB_PATH)
if results.empty:
    st.warning("No results yet. Run `crashgap ingest` and `crashgap analyze` first.")
    st.stop()

core = results[results.estimand.str.endswith("frontal_core")].iloc[0]

st.metric(
    label="Within-crash relative fatality risk, female vs male",
    value=f"{core.point:.2f}×",
    delta=f"95% CI {core.ci_lo:.2f}-{core.ci_hi:.2f}, crash-clustered bootstrap",
    delta_color="off",
)
st.markdown(
    f"In belted front-outboard mixed-sex pairs riding in the **same vehicle** in a "
    f"fatal frontal crash (FARS {core.fars_years}), the woman died and the man "
    f"survived **{core.point:.2f}×** as often as the reverse. "
    f"{int(core.n_pairs):,} mixed-sex pairs in {int(core.n_crashes):,} crashes."
)

st.subheader("Read this before quoting it")
st.markdown("""
- **This is a within-crash relative fatality risk given a fatal frontal crash — never a population rate.**
  FARS records fatal crashes only; every crash here already killed someone.
- **Not severity-adjusted beyond what the shared vehicle controls.** Same vehicle, same impact,
  same crash — but seat position (driver vs right front) and age differences within the pair remain.
- **The literature finds the gap real and shrinking in newer vehicles** (Atwood, Noh & Craig
  2023, on FARS 1975-2019 with age adjustment). The bands below from this 10-year window have
  overlapping confidence intervals and no age adjustment - they can't confirm or refute that
  trend. Don't quote a trend off them.
- **Coded fields are coarse.** Belt use and injury severity are police-coded (KABCO), not
  crash-investigation measurements. Severity-adjusted odds ratios need CISS - that is the next rung,
  not this tile.
""")

st.subheader("By vehicle model year (no trend claim)")
bands = load_bands(DB_PATH)
st.dataframe(bands.rename(columns={
    "mod_year_band": "vehicle model years", "f_only": "she died, he survived",
    "m_only": "he died, she survived", "n_pairs": "mixed-sex pairs", "rr": "RR",
    "ci_lo": "CI low", "ci_hi": "CI high",
}), hide_index=True)

st.subheader("Sensitivity: frontal definition")
sens = results[["estimand", "point", "ci_lo", "ci_hi", "n_pairs", "n_crashes"]].copy()
sens["estimand"] = sens.estimand.str.replace("fars_doublepair_fatality_f_vs_m_frontal_", "")
st.dataframe(sens.rename(columns={"estimand": "frontal definition"}), hide_index=True)

st.subheader("Method")
st.markdown("""
Evans double-pair design: only vehicles carrying **exactly one eligible male and one eligible
female** (both belted, front outboard, age 16-96, light vehicle) enter. Crash severity, delta-V,
vehicle and impact direction are held physically constant within the vehicle. The estimate is the
ratio of discordant outcomes. Frontal = IMPACT1 in {11, 12, 1} (Forman 2019); wider fan and
head-on cuts shown as sensitivity variants. Confidence intervals resample whole crashes, because
occupants of one crash are not independent observations.

The finding itself is settled science (Evans 1988; Bose 2011; Forman 2019; Atwood, Noh & Craig
2023). CrashGap adds the live, versioned, queryable layer: every number on this page sits in a
`results` table stamped with run timestamp, git commit, FARS years and the full serialized cohort
definition.
""")

with st.expander("Cohort definition (serialized, as stored with the result)"):
    st.json(json.loads(core.cohort_def))

st.caption(f"FARS {core.fars_years} · run {core.run_ts} · commit {core.git_commit} · "
           "Data: NHTSA FARS (US public domain).")
