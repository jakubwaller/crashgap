"""CrashGap dashboard.

Leads with the seat/sex decomposition, not the pooled ratio. The pooled
double-pair number is seat-confounded - seat position is the one attribute
that differs within every pair and it is ~73/27 collinear with sex - so
showing it alone would be the exact number a skeptic dismantles in one query.

Two windows, deliberately separate: the HEADLINE sections read the modern
decade (the newest MODERN_WINDOW calendar years ingested), so "does being
female raise fatality risk in the same crash today" never quietly absorbs
1990s vehicles as more history is ingested; the TREND and same-sex sections
read the full ingested span, where the older model-year bands actually have
power. Every number states which window it comes from.

The page leads with charts and short claims; the full prose sits in
expanders on the same page, one click away, never in a tooltip.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from crashgap.analysis import POWER_FLOOR, VEHICLE_AGE_MAX, modern_span
from crashgap.severity_codebook import SEV_EVENT_FLOOR

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
def db_span(db_path: str) -> tuple[int, int] | None:
    """Full ingested (min_year, max_year) span, or None on an empty DB."""
    if not Path(db_path).exists():
        return None
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT MIN(year), MAX(year) FROM person WHERE source='fars'").fetchone()
    finally:
        conn.close()
    if row is None or row[0] is None:
        return None
    return int(row[0]), int(row[1])


@st.cache_data(ttl=3600)
def load_results(db_path: str, fars_years: str) -> pd.DataFrame:
    """Latest run of each estimand family FOR THE GIVEN WINDOW. v0
    (`fars_doublepair_*`) and v1 (`fars_condlogit_*`/`fars_samesex_*`) rows
    are written by separate analyze()/analyze_v1() calls under their own
    run_ts, and each window (modern headline vs full trend span) is its own
    run of each - so the latest run is taken per (family, fars_years), never
    globally: a global MAX(run_ts) would surface whichever window happened to
    run last."""
    if not Path(db_path).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        v0 = pd.read_sql_query(
            "SELECT * FROM results WHERE estimand LIKE 'fars_doublepair%' "
            "AND fars_years = :w AND run_ts = (SELECT MAX(run_ts) FROM results "
            "WHERE estimand LIKE 'fars_doublepair%' AND fars_years = :w)",
            conn, params={"w": fars_years})
        v1 = pd.read_sql_query(
            "SELECT * FROM results WHERE estimand NOT LIKE 'fars_doublepair%' "
            "AND fars_years = :w AND run_ts = (SELECT MAX(run_ts) FROM results "
            "WHERE estimand NOT LIKE 'fars_doublepair%' AND fars_years = :w)",
            conn, params={"w": fars_years})
        return pd.concat([v0, v1], ignore_index=True)
    finally:
        conn.close()


@st.cache_data(ttl=3600)
def load_strata(db_path: str, span: tuple[int, int]) -> dict:
    from crashgap.analysis import decompose
    conn = sqlite3.connect(db_path)
    try:
        return {v: decompose(conn, span, v, reps=500) for v in ("core", "wide", "headon")}
    finally:
        conn.close()


@st.cache_data(ttl=3600)
def load_bands(db_path: str, span: tuple[int, int]) -> pd.DataFrame:
    from crashgap.analysis import by_model_year_band
    conn = sqlite3.connect(db_path)
    try:
        return by_model_year_band(conn, span, reps=500)
    finally:
        conn.close()


def _discordant_count(cohort_def: str) -> int | None:
    n = json.loads(cohort_def).get("n_discordant_pairs")
    return int(n) if n is not None else None


def _band_rows(results: pd.DataFrame, frontal: str, suffix: str) -> pd.DataFrame:
    """`fars_condlogit_sex_effect_frontal_{frontal}_band_{label}{suffix}` rows
    with a mod_year_band column, for exactly the given sensitivity suffix
    ('' = the default pooled-nuisance model)."""
    prefix = f"{CONDLOGIT_SEX_BAND}{frontal}_band_"
    rows = results[results.estimand.str.startswith(prefix)].copy()
    if rows.empty:
        return rows
    # bracket access throughout: "tail" would collide with DataFrame.tail()
    rows["band_tail"] = rows.estimand.str.slice(len(prefix))
    if suffix:
        rows = rows[rows["band_tail"].str.endswith(suffix)].copy()
        rows["mod_year_band"] = rows["band_tail"].str.removesuffix(suffix)
    else:
        rows = rows[~rows["band_tail"].str.contains("_")].copy()
        rows["mod_year_band"] = rows["band_tail"]
    rows["n_discordant_pairs"] = rows.cohort_def.apply(_discordant_count)
    return rows.drop(columns="band_tail")


@st.cache_data(ttl=3600)
def load_condlogit_bands(db_path: str, fars_years: str, span: tuple[int, int],
                         frontal: str = "core") -> pd.DataFrame:
    """Naive (unadjusted, seat-confounded) vs age-adjusted per-band female OR,
    side by side - the direct answer to the trend question (design doc
    section 5). Age-adjusted numbers come from the default pooled-nuisance
    band model; the sensitivity variants are rendered separately by
    load_band_specs below."""
    naive = load_bands(db_path, span)
    results = load_results(db_path, fars_years)
    adj = _band_rows(results, frontal, "")
    if adj.empty:
        naive["adj_point"] = None
        naive["adj_ci_lo"] = None
        naive["adj_ci_hi"] = None
        naive["n_discordant_pairs"] = None
        naive["headline_safe"] = True
        return naive
    adj = adj.rename(columns={"point": "adj_point", "ci_lo": "adj_ci_lo", "ci_hi": "adj_ci_hi"})
    merged = naive.merge(
        adj[["mod_year_band", "adj_point", "adj_ci_lo", "adj_ci_hi", "n_discordant_pairs"]],
        on="mod_year_band", how="left")
    merged["headline_safe"] = merged.n_discordant_pairs.isna() | (
        merged.n_discordant_pairs >= POWER_FLOOR)
    return merged


@st.cache_data(ttl=3600)
def load_band_specs(db_path: str, fars_years: str, frontal: str = "core") -> pd.DataFrame:
    """The specification-check table: the per-band female OR under the default
    pooled-nuisance model, the separate-nuisance refit, and the
    vehicle-age-capped refit, side by side. A band-level claim that does not
    hold across all three columns is a claim about a modelling assumption,
    not about the data."""
    results = load_results(db_path, fars_years)
    specs = {
        "default (pooled nuisance)": "",
        "separate nuisance": "_separatenuisance",
        f"vehicle age ≤ {VEHICLE_AGE_MAX}": f"_vehage{VEHICLE_AGE_MAX}",
    }
    out = None
    for name, suffix in specs.items():
        rows = _band_rows(results, frontal, suffix)
        if rows.empty:
            continue
        rows = rows[["mod_year_band", "point", "ci_lo", "ci_hi", "n_discordant_pairs"]].copy()
        rows[name] = rows.apply(
            lambda r: f"{r.point:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
            if pd.notna(r.point) else "—", axis=1)
        low = rows["n_discordant_pairs"].fillna(0) < POWER_FLOOR
        rows.loc[low & rows[name].ne("—"), name] += " (low power)"
        rows = rows[["mod_year_band", name]]
        out = rows if out is None else out.merge(rows, on="mod_year_band", how="outer")
    return out if out is not None else pd.DataFrame()


@st.cache_data(ttl=3600)
def load_condlogit_trend(db_path: str, fars_years: str,
                         frontal: str = "core") -> pd.Series | None:
    """Continuous model-year trend slope (log-odds per decade), the
    robustness cross-check against the discrete per-band pattern above."""
    results = load_results(db_path, fars_years)
    hits = results[results.estimand == f"{CONDLOGIT_TREND}{frontal}"]
    return hits.iloc[0] if not hits.empty else None


@st.cache_data(ttl=3600)
def load_samesex_interaction(db_path: str, fars_years: str,
                             frontal: str = "core") -> dict:
    """Same-sex passenger-vs-driver seat baselines (male, female), raw and
    age-adjusted, the two interaction log-ratios, and the age-comparable
    fragility refit - never shown without the baselines, per the repo's
    existing 'never show the confounded number alone' convention (design doc
    section 8). The age-adjusted rows are the headline versions; the
    age-comparable row is the check on how much of them the age model
    carries."""
    results = load_results(db_path, fars_years)
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
    name = f"fars_samesex_seatsex_interaction_ageadj_agecomparable_frontal_{frontal}"
    out["interaction_agecomparable"] = row.loc[name] if name in row.index else None
    return out


@st.cache_data(ttl=3600)
def load_severity(db_path: str) -> pd.DataFrame:
    """Latest analyze_v2 run: every {ciss,nass}_svylogit_* row, with the
    cohort_def JSON parsed into columns the section below reads. v2 rows
    carry their own source year spans in fars_years, so they never collide
    with the FARS window loads above."""
    if not Path(db_path).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        rows = pd.read_sql_query(
            "SELECT * FROM results WHERE estimand LIKE '%_svylogit_%' "
            "AND run_ts = (SELECT MAX(run_ts) FROM results "
            "WHERE estimand LIKE '%_svylogit_%')", conn)
    finally:
        conn.close()
    if rows.empty:
        return rows
    info = rows["cohort_def"].apply(json.loads)
    for key in ("n_events", "df", "kish_neff", "weight_max_over_median",
                "n_obs", "share_dv_known", "ais_revision"):
        rows[key] = info.apply(lambda d, k=key: d.get(k))
    return rows


def _sev_row(sev: pd.DataFrame, estimand: str) -> pd.Series | None:
    hits = sev[sev.estimand == estimand]
    return hits.iloc[0] if not hits.empty else None


def _sev_ok(row: pd.Series | None) -> bool:
    """v2's power floor (severity_codebook.SEV_EVENT_FLOOR unweighted outcome
    events, plus at least one design df) - the same rule
    analysis.headline_safe_mask enforces on these rows; mirrors
    _row_headline_safe for the pair channels."""
    return (row is not None and pd.notna(row.point)
            and (row.n_events or 0) >= SEV_EVENT_FLOOR and (row.df or 0) >= 1)


def _sev_fmt(row: pd.Series | None) -> str:
    """Benchmark-table cell: every ledger row renders, but one below the
    power floor is labeled so it is never read at full weight."""
    if row is None or pd.isna(row.point):
        return "—"
    cell = f"{row.point:.2f} [{row.ci_lo:.2f}, {row.ci_hi:.2f}]"
    return cell if _sev_ok(row) else cell + " (low power)"


def piecewise_agreement_text(condlogit_bands: pd.DataFrame, fars_years: str) -> str:
    """Compares the quadratic default per-band female OR against the
    `_agepiecewise` sensitivity variant and states, in one sentence, whether
    they agree — the dynamic half of the age-functional-form caveat rather
    than a static claim that could go stale."""
    results = load_results(DB_PATH, fars_years)
    pw = _band_rows(results, "core", "_agepiecewise")
    if pw.empty or condlogit_bands.empty:
        return "the piecewise-linear check could not be compared in this run (no rows available)."
    merged = condlogit_bands.merge(pw[["mod_year_band", "point"]], on="mod_year_band", how="inner")
    merged = merged.dropna(subset=["adj_point", "point"])
    if merged.empty:
        return "the piecewise-linear check could not be compared in this run (no overlapping bands)."
    max_diff = float(((merged["point"] - merged["adj_point"]).abs() / merged["adj_point"]).max())
    if max_diff < 0.02:
        return (f"it agrees closely with the quadratic default (largest per-band point-estimate "
                f"difference: {max_diff * 100:.1f}%).")
    return (f"it disagrees materially with the quadratic default in at least one band (largest "
            f"per-band point-estimate difference: {max_diff * 100:.1f}%), so treat the trend "
            f"conclusion as an open risk rather than settled.")


def pct(x: float | None) -> str:
    return f"{(x - 1) * 100:+.1f}%" if x else "n/a"


# Chart colors: one data hue plus a neutral for reference lines, benchmarks and
# low-power de-emphasis. Identity never rides on color alone - low-power marks
# are labeled, benchmarks get their own shape and a legend.
BLUE = "#3987e5"
GRAY = "#898781"


def split_chart(core) -> alt.Chart:
    """The headline decomposition as a picture: her-vs-his death ratio in the
    two seat configurations, with the parity line. The flip across 1.0 is the
    seat effect."""
    df = pd.DataFrame([
        {"configuration": "she rides passenger", "ratio": core.rr_she_passenger,
         "pairs": core.n_pairs_she_passenger},
        {"configuration": "she drives", "ratio": core.rr_she_drives,
         "pairs": core.n_pairs_she_drives},
    ])
    xmax = max(1.3, float(df["ratio"].max()) * 1.15)
    bars = alt.Chart(df).mark_bar(color=BLUE, size=26, cornerRadiusEnd=4).encode(
        x=alt.X("ratio:Q", title="her deaths : his deaths",
                scale=alt.Scale(domain=[0, xmax])),
        y=alt.Y("configuration:N", title=None, sort=None),
        tooltip=[alt.Tooltip("configuration:N"),
                 alt.Tooltip("ratio:Q", format=".3f"),
                 alt.Tooltip("pairs:Q", format=",")])
    labels = alt.Chart(df).mark_text(align="left", dx=6, color=GRAY).encode(
        x="ratio:Q", y=alt.Y("configuration:N", sort=None),
        text=alt.Text("ratio:Q", format=".2f"))
    rule = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(
        color=GRAY, strokeWidth=2, strokeDash=[4, 3]).encode(x="x:Q")
    return (bars + labels + rule).properties(height=130)


def trend_chart(bands: pd.DataFrame) -> alt.Chart | None:
    """Age-adjusted female OR per model-year band with 95% CI. Low-power bands
    render gray (the same headline_safe rule as the table), never dropped."""
    df = bands.dropna(subset=["adj_point"]).copy()
    if df.empty:
        return None
    df["power"] = df["headline_safe"].map(lambda ok: "ok" if ok else "low power")
    color = alt.Color(
        "power:N", title=None,
        scale=alt.Scale(domain=["ok", "low power"], range=[BLUE, GRAY]),
        legend=None if (df["power"] == "ok").all() else alt.Legend(orient="top"))
    tooltip = [alt.Tooltip("mod_year_band:N", title="model years"),
               alt.Tooltip("adj_point:Q", format=".2f", title="female OR"),
               alt.Tooltip("adj_ci_lo:Q", format=".2f", title="CI low"),
               alt.Tooltip("adj_ci_hi:Q", format=".2f", title="CI high"),
               alt.Tooltip("n_discordant_pairs:Q", format=",", title="discordant pairs"),
               alt.Tooltip("power:N")]
    x = alt.X("mod_year_band:N", title="vehicle model years", sort=None,
              axis=alt.Axis(labelAngle=0))
    base = alt.Chart(df)
    ci = base.mark_rule(strokeWidth=2).encode(
        x=x, y=alt.Y("adj_ci_lo:Q", title="age-adjusted female odds ratio",
                     scale=alt.Scale(zero=False)),
        y2="adj_ci_hi:Q", color=color, tooltip=tooltip)
    pts = base.mark_point(filled=True, size=90).encode(
        x=x, y="adj_point:Q", color=color, tooltip=tooltip)
    labels = base.mark_text(align="left", dx=10, color=GRAY).encode(
        x=x, y="adj_point:Q", text=alt.Text("adj_point:Q", format=".2f"))
    rule = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(
        color=GRAY, strokeWidth=2, strokeDash=[4, 3]).encode(y="y:Q")
    return (ci + pts + labels + rule).properties(height=260)


def severity_chart(sev: pd.DataFrame) -> alt.Chart | None:
    """The delta-V-adjusted severity ORs with CIs, against the published
    benchmarks as gray diamonds. A diamond appears only where the eras match
    (Craig 2024 measured MAIS 2+ in CISS, Bose 2011 measured MAIS 3+ in
    NASS-CDS); the other two cells have no like-for-like published number,
    per docs/v2-design.md. Below-floor cells fade and say so in the
    tooltip."""
    cells = {("ciss", "mais2plus"): "CISS · MAIS 2+",
             ("ciss", "mais3plus"): "CISS · MAIS 3+",
             ("nass", "mais2plus"): "NASS · MAIS 2+",
             ("nass", "mais3plus"): "NASS · MAIS 3+"}
    benchmarks = {("ciss", "mais2plus"): ("Craig 2024 (CISS era)", 1.75),
                  ("nass", "mais3plus"): ("Bose 2011 (NASS era)", 1.47)}
    est_rows, bench_rows = [], []
    for (source, outcome), label in cells.items():
        r = _sev_row(sev, f"{source}_svylogit_female_or_{outcome}_frontal_dv")
        if r is None or pd.isna(r.point):
            continue
        est_rows.append({"cell": label, "point": r.point, "ci_lo": r.ci_lo,
                         "ci_hi": r.ci_hi,
                         "power": "ok" if _sev_ok(r) else "low power"})
        if (source, outcome) in benchmarks:
            name, val = benchmarks[(source, outcome)]
            bench_rows.append({"cell": label, "point": val, "benchmark": name})
    if not est_rows:
        return None
    order = [label for label in cells.values()
             if label in {r["cell"] for r in est_rows}]
    series = alt.Scale(domain=["this analysis (ΔV-adjusted)", "published benchmark"],
                       range=[BLUE, GRAY])
    shapes = alt.Scale(domain=["this analysis (ΔV-adjusted)", "published benchmark"],
                       range=["circle", "diamond"])
    est = pd.DataFrame(est_rows)
    est["series"] = "this analysis (ΔV-adjusted)"
    # explicit columns so an empty benchmark layer still compiles
    bench = pd.DataFrame(bench_rows, columns=["cell", "point", "benchmark"])
    bench["series"] = "published benchmark"
    faded = alt.condition(alt.datum.power == "low power",
                          alt.value(0.45), alt.value(1.0))
    y = alt.Y("cell:N", title=None, sort=order)
    ci = alt.Chart(est).mark_rule(strokeWidth=2, color=BLUE).encode(
        x=alt.X("ci_lo:Q", title="female odds ratio", scale=alt.Scale(zero=False)),
        x2="ci_hi:Q", y=y, opacity=faded)
    pts = alt.Chart(est).mark_point(filled=True, size=90).encode(
        x="point:Q", y=y, opacity=faded,
        color=alt.Color("series:N", title=None, scale=series,
                        legend=alt.Legend(orient="top")),
        shape=alt.Shape("series:N", title=None, scale=shapes),
        tooltip=[alt.Tooltip("cell:N"), alt.Tooltip("point:Q", format=".2f"),
                 alt.Tooltip("ci_lo:Q", format=".2f", title="CI low"),
                 alt.Tooltip("ci_hi:Q", format=".2f", title="CI high"),
                 alt.Tooltip("power:N")])
    marks = alt.Chart(bench).mark_point(filled=True, size=110).encode(
        x="point:Q", y=y,
        color=alt.Color("series:N", title=None, scale=series,
                        legend=alt.Legend(orient="top")),
        shape=alt.Shape("series:N", title=None, scale=shapes),
        tooltip=[alt.Tooltip("cell:N"), alt.Tooltip("benchmark:N"),
                 alt.Tooltip("point:Q", format=".2f")])
    rule = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(
        color=GRAY, strokeWidth=2, strokeDash=[4, 3]).encode(x="x:Q")
    return (ci + pts + marks + rule).properties(height=230)


st.title("CrashGap")
st.caption("Female-vs-male fatality risk in the same crash, computed live from "
           "public-domain NHTSA FARS data, with the seat-position confound "
           "separated out.")

span = db_span(DB_PATH)
if span is None:
    st.warning("No data yet. Run `crashgap ingest` and `crashgap analyze` first.")
    st.stop()
mspan = modern_span(span)
FULL = f"{span[0]}-{span[1]}"
MODERN = f"{mspan[0]}-{mspan[1]}"

results = load_results(DB_PATH, MODERN)
if results.empty:
    st.warning("No results yet. Run `crashgap analyze` and `crashgap analyze-v1` first.")
    st.stop()

row = results.set_index("estimand")
sex, seat, pooled = row.loc[SEX + "core"], row.loc[SEAT + "core"], row.loc[POOLED + "core"]
strata = load_strata(DB_PATH, mspan)

st.markdown(f"Headline: FARS {MODERN}, the modern decade. The trend and same-sex sections "
            f"use the full span, FARS {FULL}. Every number says which window it uses.")

left, right = st.columns(2)
left.metric("Being female (seat-balanced)", pct(sex.point),
            delta=f"95% CI {pct(sex.ci_lo)} to {pct(sex.ci_hi)}", delta_color="off")
right.metric("Sitting in the right-front seat", pct(seat.point),
             delta=f"95% CI {pct(seat.ci_lo)} to {pct(seat.ci_hi)}", delta_color="off")

st.markdown(f"""
Among belted front-seat mixed-sex pairs riding in the **same vehicle** in a fatal frontal crash
(FARS {sex.fars_years}, {int(sex.n_pairs):,} pairs in {int(sex.n_crashes):,} crashes), the seat
effect is larger than the sex effect. Sitting right-front rather than driving raises the relative
fatality risk by **{pct(seat.point)}** and that is statistically significant. Being female raises
it by **{pct(sex.point)}**, and in this modern window that is not distinguishable from zero.
""")

st.error(f"""Pooling the pairs without splitting by seat gives a female-vs-male ratio of
**{pooled.point:.2f}×**, which reads like a {pct(pooled.point)} female penalty. Most of it is
the seat effect: women are the passenger in {strata['core'].n_pairs_she_passenger:,} of the
{strata['core'].n_pairs:,} pairs, so "female" and "right-front" are ~73/27 collinear.""")

core = strata["core"]
st.markdown(f"**Who died, her vs him, by who drives (FARS {MODERN})**")
st.altair_chart(split_chart(core), width="stretch")
st.caption("The ratio flips below 1.0 when she drives. That flip is the seat effect.")

with st.expander("The raw counts and the frontal cuts"):
    st.dataframe(pd.DataFrame([
        {"configuration": "she is the passenger, he drives",
         "pairs": core.n_pairs_she_passenger,
         "ratio (she died : he died)": core.rr_she_passenger},
        {"configuration": "she drives, he is the passenger",
         "pairs": core.n_pairs_she_drives,
         "ratio (she died : he died)": core.rr_she_drives},
    ]), hide_index=True)
    st.caption("If sex alone drove the outcome these two ratios would agree. They don't: the "
               "ratio flips below 1.0 when she is the one driving, which is the seat effect "
               "showing through. The geometric mean of the two cancels seat and isolates sex; "
               "their geometric ratio cancels sex and isolates seat (Evans' double-pair "
               "decomposition).")

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
    _headon = strata["headon"]
    st.warning(f"""The cuts disagree, and the disagreement carries information. In the two broad
frontal cuts the female effect is small and not significant. In the strict head-on cut
(`MAN_COLL=2`) it is **{pct(_headon.sex_effect)}
({'significant' if _headon.sex_is_significant else 'not significant'})**, and the seat effect
*reverses*, with the driver faring worse. That reversal is physically plausible: head-on
collisions are predominantly offset toward the driver's side, which is why the driver-side
small-overlap crash test existed for six years before a passenger-side version was added.
Depending on crash geometry the female effect sits somewhere between low single digits and low
double digits, and this window cannot pin it down further. Quoting only the head-on row, or
only the core row, is cherry-picking.""")

st.subheader("Does the gap shrink as cars get newer?")
condlogit_bands = load_condlogit_bands(DB_PATH, FULL, span)
st.markdown(f"The published trend says yes. This data (FARS {FULL}) confirms the drop from "
            f"pre-2000 vehicles to 2000s vehicles under every specification. After the 2000s "
            f"nothing declines further, and whether the newest band sits above the 2000s "
            f"level depends on a modelling choice.")
_tchart = trend_chart(condlogit_bands)
if _tchart is not None:
    st.altair_chart(_tchart, width="stretch")
    st.caption(f"Age-adjusted female odds ratio by vehicle model year, with 95% CI "
               f"(FARS {FULL}, default specification). Vehicle model years, not calendar "
               f"years. Gray marks sit below the {POWER_FLOOR}-discordant-pair power floor.")

with st.expander("Trend details, specification checks and the naive bands"):
    st.markdown(f"""
The direct test of the published shrinking trend (Atwood, Noh & Craig 2023: a 19.9% female
penalty in the oldest vehicles falling to 5.8% in the newest): a within-vehicle differenced
conditional logit over the full {FULL} span that holds seat position and occupant age constant
(`x_seat`, `x_age`, `x_age_curv` shared, one female coefficient per model-year band), with two
sensitivity refits, one with nuisance parameters per band and one with vehicles capped at
{VEHICLE_AGE_MAX} years old at crash.

What the full window settles: pre-2000 vehicles carry a clearly higher age-adjusted female
penalty (~+23% to +28%) than 2000s vehicles (~+5% to +11%). That drop is significant with
disjoint CIs under every specification below, and it reproduces the direction of the published
decline. After the 2000s band the reproduction stops. Nothing declines further under any
specification: the 2010–2016 band sits above the 2000s level in all three, and the newest band
does too except under the separate-nuisance refit, where the two are indistinguishable (1.10 vs
1.11). The default fit even reads the newest band significantly *above* the 2000s band (a Wald
contrast on the pooled-nuisance model's own covariance gives p ≈ 0.03). That contrast does not
survive the separate-nuisance refit, and no post-2000 contrast has disjoint CIs under any
specification, so it stays a spec-dependent hint of a reversal and is not counted as a finding.
The newest band still flips with the nuisance-pooling choice (1.26 pooled vs 1.10 separate; the
age-adjusted seat effect varies by band, roughly 1.0 in the older bands vs 1.34 [1.17–1.53] in
the newest, so the data reject pooling exactly where it matters most). The continuous slope
over 55 model years is −0.025 log-odds per decade (95% CI −0.062 to +0.013), consistent with a
slow decline and with zero. In sum, the historical decline into the 2000s is real in this data,
and a continued decline into the newest vehicles is not found. That may reflect the extra
2020–2024 calendar years this window adds beyond the published study's 2019 endpoint, a period
whose crash mix shifted sharply, or a genuine plateau. This design cannot separate those.
""")

    def _grey_low_power(row: pd.Series) -> list[str]:
        if row["headline safe"] == "low power":
            return ["color: #888888; background-color: rgba(128,128,128,0.12)"] * len(row)
        return [""] * len(row)

    band_display = condlogit_bands.rename(columns={
        "mod_year_band": "vehicle model years", "n_pairs": "pairs",
        "f_only": "she died, he didn't", "m_only": "he died, she didn't",
        "rr": "naive RR (unadjusted, pooled)",
        "ci_lo": "naive CI low", "ci_hi": "naive CI high",
        "adj_point": "age-adjusted female OR", "adj_ci_lo": "adjusted CI low",
        "adj_ci_hi": "adjusted CI high", "n_discordant_pairs": "discordant pairs",
    })

    # Reuse the headline_safe column load_condlogit_bands already computed from
    # POWER_FLOOR, rather than re-deriving the same rule here with a second
    # literal threshold that could drift out of sync with it.
    band_display["headline safe"] = band_display["headline_safe"].map(
        lambda ok: "ok" if ok else "low power")
    cols = ["vehicle model years", "pairs", "she died, he didn't", "he died, she didn't",
            "naive RR (unadjusted, pooled)", "naive CI low", "naive CI high",
            "age-adjusted female OR", "adjusted CI low", "adjusted CI high",
            "discordant pairs", "headline safe"]
    st.dataframe(band_display[cols].style.apply(_grey_low_power, axis=1), hide_index=True)
    st.caption(f"Rows flagged **low power** carry fewer than the {POWER_FLOOR}-discordant-pair "
               "floor this project requires before headlining a point estimate (design doc "
               "section 5). They stay in the table, greyed out, and never carry the same "
               "weight as an adequately powered band. Read them as directional only. The "
               "naive columns use the pooled (seat-confounded) ratio, unadjusted for occupant "
               "age; with women averaging ~1.4 years younger within pairs, they UNDERSTATE "
               "the female OR relative to the age-adjusted ones.")

    specs = load_band_specs(DB_PATH, FULL)
    if not specs.empty:
        st.markdown("**Specification check**: the same age-adjusted per-band female OR under "
                    "the three specifications. A band-level claim that does not hold in all "
                    "three columns depends on a modelling assumption rather than on the data:")
        st.dataframe(specs.rename(columns={"mod_year_band": "vehicle model years"}),
                     hide_index=True)
        st.caption(f"*Separate nuisance* refits seat/age independently inside each band "
                   f"instead of pooling them (the data reject pooling: the per-band seat "
                   f"effect has disjoint CIs). *Vehicle age ≤ {VEHICLE_AGE_MAX}* restricts "
                   f"every band to its first {VEHICLE_AGE_MAX} years on the road, putting "
                   f"bands on comparable vehicle-age support, at the price that band and "
                   f"calendar period become nearly synonymous, as in the published FARS trend "
                   f"estimates.")

    trend = load_condlogit_trend(DB_PATH, FULL)
    if trend is not None and pd.notna(trend.point):
        st.caption(f"Continuous-trend cross-check (fit separately from the band dummies, "
                   f"never jointly, since band is a coarsened function of year): "
                   f"**{trend.point:+.4f}** log-odds per decade (95% CI {trend.ci_lo:+.4f} "
                   f"to {trend.ci_hi:+.4f}). Its sign should agree with the discrete "
                   f"per-band pattern above; do not quote it as a third headline number.")

st.subheader("Is the passenger seat worse for women specifically?")
st.markdown(f"Mixed-sex pairs cannot answer this, so male-male and female-female pairs are "
            f"compared instead (FARS {FULL}).")
samesex = load_samesex_interaction(DB_PATH, FULL)
male_seat, female_seat = samesex["male"], samesex["female"]
male_adj, female_adj = samesex["male_ageadj"], samesex["female_ageadj"]
interaction = samesex["interaction"]
interaction_adj = samesex["interaction_ageadj"]
interaction_cmp = samesex["interaction_agecomparable"]


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

# The directional claim renders only when the estimate backing it exists,
# clears the power floor, and is itself significant; the design doc requires
# the null reading and the weaker-assumption note next to the number, not
# below the fold (docs/v1-design.md sections 3 and 6).
if male_ok and female_ok and interaction_ok:
    if interaction_adj.ci_lo > 0 or interaction_adj.ci_hi < 0:
        st.markdown("Age-adjusted, the passenger seat penalizes female-female pairs where it "
                    "spares male-male ones. The size depends on the age model, and this "
                    "channel rests on a weaker identifying assumption than the rest of the "
                    "page.")
    else:
        st.markdown("Age-adjusted, the seat-by-sex interaction is not distinguishable from "
                    "zero in this run. At this sample size that is not strong evidence of no "
                    "effect, and this channel rests on a weaker identifying assumption than "
                    "the rest of the page.")

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
    ix_left, ix_right = st.columns(2)
    ix_left.metric("Age-adjusted interaction: log(female seat effect / male seat effect)",
                   f"{interaction_adj.point:+.3f}",
                   delta=f"95% CI {interaction_adj.ci_lo:+.3f} to {interaction_adj.ci_hi:+.3f}",
                   delta_color="off")
    if _row_headline_safe(interaction_cmp):
        ix_right.metric("Same interaction, age-comparable pairs only (gap ≤ 10 yrs)",
                        f"{interaction_cmp.point:+.3f}",
                        delta=f"95% CI {interaction_cmp.ci_lo:+.3f} to "
                              f"{interaction_cmp.ci_hi:+.3f}",
                        delta_color="off")
        st.caption("Read the two interaction numbers together: neither alone is the "
                   "supportable range. The age-comparable refit shows how much of the "
                   "full-cohort estimate the age model carries.")
    with st.expander("Baselines, raw diagnostics and the identifying assumption"):
        st.markdown("""
A `female × seat` term cannot be recovered from mixed-sex pairs at all: within any discordant
mixed-sex pair her indicator minus his is identically 1, so that term is algebraically the seat
main effect (design doc section 6). This channel routes around that. The
right-front-passenger-vs-driver fatality odds are computed separately within male-male and
female-female discordant pairs and shown side by side, so the interaction ratio can be read
against its baselines.

The raw baselines are age-confounded and serve as diagnostics only. The two cohorts differ
sharply in who sits where at what age: female-female pairs carry a much older right-front
passenger far more often than male-male pairs do, and age is the strongest single predictor of
who dies, ahead of seat and sex. The headline numbers are therefore the age-adjusted fits
(quadratic within-pair age difference, same machinery as the mixed-sex model). The raw ORs sit
below so the size of the age correction stays visible.
""")
        if raw_ok:
            st.caption(f"Unadjusted diagnostics: male-male {male_seat.point:.3f}× "
                       f"[{male_seat.ci_lo:.3f}, {male_seat.ci_hi:.3f}], female-female "
                       f"{female_seat.point:.3f}× [{female_seat.ci_lo:.3f}, "
                       f"{female_seat.ci_hi:.3f}], raw log-ratio {interaction.point:+.3f} "
                       f"[{interaction.ci_lo:+.3f}, {interaction.ci_hi:+.3f}]. The gap between "
                       f"the raw and age-adjusted rows IS the age-composition confound. Do not "
                       f"quote the raw asymmetry as a seat or sex effect.")
        st.caption("How to read the pair of interaction numbers: on the full window the "
                   "age-adjusted interaction is positive, statistically significant, and "
                   "stable in direction across calendar eras. The passenger seat penalizes "
                   "female-female pairs where it spares male-male ones. Its *size* depends on "
                   "the age model, though: restricted to age-comparable pairs (where the "
                   "quadratic adjustment interpolates instead of extrapolating) the estimate "
                   "drops to roughly half and its CI touches zero. What this supports is a "
                   "female-specific right-front penalty of modest size, with the magnitude "
                   "uncertain between roughly +10% and +23%, carried disproportionately by "
                   "large-age-gap pairs.")
        st.caption("All of it rests on a weaker identifying assumption than every other number "
                   "on this page: same-sex pairs are not sex-paired within the same crash, so "
                   "who rides with whom could correlate with crash severity or vehicle type in "
                   "ways the mixed-pair design structurally rules out. This channel is also "
                   "the most power-starved on the page, because two-occupant fatal crashes "
                   "skew heavily mixed-sex.")
else:
    st.info(f"Same-sex interaction estimand not available, or below the {POWER_FLOOR}-discordant-"
            f"pair power floor, in this run.")

sev = load_severity(DB_PATH)
if not sev.empty:
    ciss_years = sev[sev.estimand.str.startswith("ciss_")]["fars_years"].iloc[0] \
        if (sev.estimand.str.startswith("ciss_")).any() else None
    nass_years = sev[sev.estimand.str.startswith("nass_")]["fars_years"].iloc[0] \
        if (sev.estimand.str.startswith("nass_")).any() else None
    st.subheader("The severity-adjusted gap")
    st.markdown(f"FARS only knows who died. CISS ({ciss_years or '—'}) and NASS-CDS "
                f"({nass_years or '—'}) grade injuries and reconstruct crash forces, which is "
                f"where the published numbers live. In the two cells with a like-for-like "
                f"published benchmark (CISS MAIS 2+ against Craig 2024, NASS MAIS 3+ against "
                f"Bose 2011), the delta-V-adjusted estimate lands near the published number "
                f"without tuning. The other two cells have no era-matched benchmark.")
    _schart = severity_chart(sev)
    if _schart is not None:
        st.altair_chart(_schart, width="stretch")
        st.caption("Female odds ratio for a moderate-or-worse (MAIS 2+) and serious-or-worse "
                   "(MAIS 3+) injury, belted front-outboard adults in frontal light-vehicle "
                   "crashes, delta-V-adjusted, with 95% CI. Faded marks sit below the power "
                   "floor. Diamonds mark each published benchmark in its own era's data set "
                   "only; the full cross table, era mismatches included, is in the expander "
                   "below.")
    sev_expander = st.expander("The benchmark table, weight diagnostics and AIS sensitivity")
    sev_expander.markdown("""
FARS knows who died and nothing more, so everything above is a *fatality* contrast within fatal
crashes. This rung switches data sets: CISS and its predecessor NASS-CDS are weighted national
samples of tow-away crashes with hospital-grade AIS injury coding and reconstructed delta-V.
The estimand is the female odds ratio for **MAIS 2+** (the quantity in Craig 2024, published
~1.75) and **MAIS 3+** (Bose 2011, published ~1.47) among belted front-outboard adults in
frontal light-vehicle crashes: a design-weighted logistic with Taylor-linearized,
PSU-clustered, stratum-aware standard errors and t-based CIs on the design's own degrees of
freedom (15–28 here, so the intervals come out wide rather than optimistic).

Three covariate tiers per estimate: **base** (age, seat, model-year band), **+ΔV** (adds
delta-V and its square, on the subset where reconstruction succeeded), **+ΔV+anthro** (adds
height and BMI). Height and BMI partly *mediate* a sex effect, so the third tier answers what
remains net of body size; it does not test whether the gap is real.
""")
    bench = []
    for source, outcome, published in (
            ("ciss", "mais2plus", "Craig 2024: ~1.75 (MAIS 2+, CISS era)"),
            ("ciss", "mais3plus", "Bose 2011: 1.47 (MAIS 3+, NASS era)"),
            ("nass", "mais2plus", "Craig 2024: ~1.75 (MAIS 2+, CISS era)"),
            ("nass", "mais3plus", "Bose 2011: 1.47 (MAIS 3+, NASS era)")):
        prefix = f"{source}_svylogit_female_or_{outcome}_frontal_"
        base = _sev_row(sev, prefix + "base")
        bench.append({
            "source": f"{source.upper()} ({base.fars_years if base is not None else '—'})",
            "outcome": {"mais2plus": "MAIS 2+", "mais3plus": "MAIS 3+"}[outcome],
            "base OR [95% CI]": _sev_fmt(base),
            "+ΔV": _sev_fmt(_sev_row(sev, prefix + "dv")),
            "+ΔV +height/BMI": _sev_fmt(_sev_row(sev, prefix + "dvanthro")),
            "weights trimmed (p95)": _sev_fmt(_sev_row(sev, prefix + "base_wtrim95")),
            "published benchmark": published,
        })
    sev_expander.dataframe(pd.DataFrame(bench), hide_index=True)

    ciss_base = _sev_row(sev, "ciss_svylogit_female_or_mais2plus_frontal_base")
    nass_base = _sev_row(sev, "nass_svylogit_female_or_mais3plus_frontal_base")
    ais08 = _sev_row(sev, "nass_svylogit_female_or_mais3plus_frontal_base_ais08")
    ais90dual = _sev_row(sev, "nass_svylogit_female_or_mais3plus_frontal_base_ais90dual")
    diag_bits = []
    for label, row in (("CISS", ciss_base), ("NASS", nass_base)):
        if row is not None and pd.notna(row.point):
            diag_bits.append(
                f"{label}: n={int(row.n_obs):,}, events={int(row.n_events):,}, "
                f"design df={int(row.df)}, Kish effective n={row.kish_neff:,.0f}, "
                f"max/median weight={row.weight_max_over_median:.0f}×")
    if diag_bits:
        sev_expander.caption("**Weight diagnostics** (this makes the Viano 2025 "
                             "weight-instability critique measurable): "
                             + "; ".join(diag_bits) + ". A max/median weight ratio in the "
                             "tens or hundreds means single cases can move a point estimate, "
                             "which is why the trimmed-weights column and the design-df CIs "
                             "ship next to every number.")
    if ais08 is not None and pd.notna(ais08.point):
        pair = (f"the SAME {ais08.fars_years} dual-coded cohort grades "
                f"{_sev_fmt(ais90dual)} under AIS90 and {_sev_fmt(ais08)} under AIS2008"
                if ais90dual is not None and pd.notna(ais90dual.point)
                else f"the {ais08.fars_years} dual-coded cohort grades {_sev_fmt(ais08)} "
                     f"under AIS2008")
        sev_expander.caption(f"**AIS revision sensitivity**: NASS 2010+ dual-codes injuries "
                             f"in AIS90 and AIS2008, and for MAIS 3+ {pair}. That is a "
                             f"same-cohort pair, so the difference is the injury *coding* "
                             f"itself rather than an era or cohort shift (comparing either "
                             f"against the full-window base row would conflate the two). The "
                             f"revision boundary sits between the NASS-era and CISS-era "
                             f"benchmarks, so part of any Bose-vs-Craig gap comes from coding "
                             f"rather than crash physics.")
    sev_expander.markdown("""
The delta-V-adjusted rows land where the literature lands: women's frontal crashes carry lower
delta-V on average, so the base tier *understates* the conditional injury odds and the +ΔV tier
raises every estimate, the same direction every published severity analysis reports. The gap
does not evaporate under the body-size tier: adding height and BMI leaves the NASS MAIS 3+
estimate unchanged, raises both MAIS 2+ estimates, and moves only the CISS MAIS 3+ cell
downward (the one cell that is not statistically significant in either tier), so "smaller
occupants get hurt more" does not absorb the female coefficient here. Against that stand the
two measured fragilities: single cases carry weights hundreds of times the median (trimming
moves points visibly, though not their direction), and the AIS revision alone moves the NASS
MAIS 3+ estimate materially on the dual-coded subset. What this table supports: the
severity-adjusted female odds ratio in frontal crashes sits roughly in the 1.4–2.0 band the
published literature reports, the delta-V-adjusted estimates in its upper half, with every
fragility this project could measure printed beside it.

Read the severity numbers with these in hand:

- **Complete-case cohorts.** Occupants without a completed injury workup (a large share of NASS
  occupant rows) drop out, as in every published analysis of these files.
- **Delta-V missingness is selective.** Reconstruction fails more often in the worst crashes.
  The +ΔV tiers run on the subset where it succeeded (share recorded in each row's cohort
  definition), so the base-vs-ΔV comparison is partly a cohort change on top of an adjustment.
- **This is not a reproduction of Bose or Craig.** Frontal definitions, cohorts, AIS revisions
  and covariate sets differ in documented ways (docs/v2-design.md). The benchmark column is
  there for orientation.
- **Never pooled with FARS.** Different estimand (injury odds in tow-away crashes vs death in
  fatal crashes), different data, different design.
""")

caveats = st.expander("Read this before quoting anything here")
caveats.markdown("""
- **These are within-crash relative risks given a fatal frontal crash, never population
  rates.** FARS records fatal crashes only; every crash here already killed someone.
- **Not severity-adjusted beyond what the shared vehicle controls.** Same vehicle, same impact,
  same crash, but occupant age still differs within pairs (women here average ~1.3–1.4 years
  younger). The v0 `fars_doublepair_*` rows do not adjust for that; the v1 `fars_condlogit_*`
  and `_ageadj` rows do (quadratic within-pair age difference).
- **The published severity-adjusted gap is a different, larger quantity.** Bose (2011) reports
  MAIS 3+ odds ~1.47 from crash-investigation data with injury grading and survey weights. That
  quantity is measured in the severity section above on CISS/NASS-CDS, a different data set and
  estimand. Do not read the FARS tiles as refuting it.
- **Coded fields are coarse.** Belt use and injury severity are police-coded (KABCO) rather
  than measured.
- **Two FARS recodes sit inside the full window.** The unknown-age sentinel changed in 2009 and
  IMPACT1 gained side-specific codes in 2010. Both are verified harmless for the core frontal
  cohort (see `codebook.py`), but the *wide* frontal variant selects a somewhat narrower crash
  set after 2010, so cross-era comparisons of that one variant carry an asterisk.
""")
caveats.markdown("**v1 model caveats**, to read next to the age-adjusted and interaction "
                 "numbers:")
caveats.markdown("""
- **Not a verified reproduction of any published regression form.** This is a standard matched-pairs
  differenced-logit (the conditional-likelihood identity is textbook), built to answer the trend and
  interaction questions on this data. It has not been checked against the exact model-year-band
  equations the literature this project benchmarks against actually uses.
- **The default trend answer pools `β_seat` and the age curve across model-year bands.** That is
  an assumption, and the data reject it (the per-band seat effect has disjoint CIs). The
  specification-check table above is therefore required reading: only band contrasts that hold
  under the separate-nuisance and vehicle-age-capped refits count as findings.
- **Model year, vehicle age and calendar period cannot all be separated.** They are linearly
  dependent (model year = period − vehicle age), and none of them varies within a pair. The
  uncapped fits mix them one way, the vehicle-age-capped fit another; where they agree (the
  pre-2000 → 2000s drop) the claim is robust to that mixing, where they disagree it is not
  identified.
- **Age enters as a closed-form quadratic on the within-pair age difference.** This will not capture
  a true age-mortality threshold (elevated risk past roughly 65–75). A piecewise-linear sensitivity
  variant (knot at 65, `_agepiecewise` rows below) is run alongside it; {piecewise_agreement}
  Neither functional form is a verified match to the true fatality-age curve.
- **The seat × sex interaction is identified entirely from the same-sex-pair channel above**, a small
  fraction of the eligible cohort (two-occupant fatal crashes skew mixed-sex) resting on a weaker
  identifying assumption than the mixed-pair design: same-sex pairs are not sex-paired within the
  same crash, so who rides with whom could correlate with crash severity or vehicle type. A null
  interaction result is not strong evidence of no effect at this sample size.
- **The raw same-sex baselines are age-confounded, so quote only the `_ageadj` rows,** with the
  `_agecomparable` refit next to them: the adjusted interaction is significant and era-stable on
  the full window, but roughly half of it is carried by large-age-gap pairs where the quadratic
  age model extrapolates furthest. The supportable range is the pair of numbers together, never
  either alone.
- **No covariate here reaches crash severity directly.** If severity composition itself shifted
  across model-year bands (e.g. newer, safer vehicles overrepresented among *survivable* fatal
  crashes), that channel survives age-adjustment untouched and could still explain part of the
  band pattern.
- **The 30-discordant-pair power floor.** Any point estimate backed by fewer than 30 discordant
  pairs is written to the results ledger and never headlined on this page. Read it as
  directional only.
""".format(piecewise_agreement=piecewise_agreement_text(condlogit_bands, FULL)))

method = st.expander("Method, windows and the conditional logit")
method.markdown(f"""
Evans double-pair design: only vehicles carrying **exactly one eligible male and one eligible
female** (both belted, front outboard, age 16–96, light vehicle) enter, which holds crash severity,
delta-V, vehicle and impact direction physically constant within the vehicle. The estimate is the
ratio of discordant outcomes. Running it separately in the two seat configurations and taking
geometric means splits it into a sex effect and a seat effect: with *f* the female multiplier and
*s* the right-front multiplier, the two configurations measure *f·s* and *f/s*, so `sqrt(product)`
recovers *f* and `sqrt(quotient)` recovers *s*.

Frontal = IMPACT1 ∈ {{11, 12, 1}} (Forman 2019); the wider fan and head-on cuts are shown above.
Confidence intervals are percentile bootstraps resampling **whole crashes**, because occupants of
one crash are not independent observations; both strata are resampled inside one replicate so the
two effects stay mutually consistent.

**Windows.** The headline sections use FARS {MODERN}, the modern decade, so the "today" claim
stays comparable as history accumulates. The trend and same-sex sections use the full ingested
span (FARS {FULL}): the pre-2010 model-year bands are nearly extinct on modern roads, so only the
full span gives them power. Every FARS code set the cohort depends on was re-verified across the
2000-vintage files before the span was widened (`codebook.py` documents the two recodes that sit
inside it).

The underlying finding is settled science (Evans 1988; Bose 2011; Forman 2019; Atwood, Noh & Craig
2023). What CrashGap adds is the live, versioned form: every number here sits in a `results`
table stamped with run timestamp, git commit, FARS years and the full serialized cohort
definition.

**v1 adds a within-vehicle differenced conditional logit** for the age-adjusted and interaction
tiles above. For a 2-occupant stratum the conditional MLE has a closed form: the stratum likelihood
depends only on the difference of the linear predictor between the two occupants, so fitting
reduces to a logistic regression of `y = 1{{female occupant died}}` on covariates defined as her
value minus his (`x_seat`, and an age difference, quadratic by default) over discordant pairs
only, with cluster-robust standard errors on `(year, st_case)`. It nests v0's decomposition above as
the special case with no age variation. The seat × sex interaction is not fit from these mixed
pairs at all (see the caveats above); it comes from the separate same-sex channel.
""")

st.subheader("How this site is made")
st.markdown("I do use AI to help me with coding and drafting texts. Everything goes through "
            "my human eyes though.")

with st.expander("Cohort definition (serialized, as stored with the result)"):
    st.json(json.loads(sex.cohort_def))

with st.expander("All estimands from the latest runs (both windows)"):
    table = pd.concat([load_results(DB_PATH, MODERN), load_results(DB_PATH, FULL)],
                      ignore_index=True).drop_duplicates(subset=["estimand", "fars_years"])
    table["n_discordant_pairs"] = table.cohort_def.apply(_discordant_count)
    st.dataframe(
        table[["estimand", "fars_years", "point", "ci_lo", "ci_hi", "n_pairs",
               "n_discordant_pairs", "n_crashes"]],
        hide_index=True)
    st.caption("`n_pairs` is the eligible cohort (incl. concordant pairs); `n_discordant_pairs` is "
               "the informative subset the fit actually used (blank for v0 rows, which don't "
               "serialize that split). A discordant-only fit is correct for the odds ratio, but a "
               "bare `n_pairs` on its own overstates effective power.")

st.caption(f"Headline: FARS {MODERN} · trend/same-sex: FARS {FULL} · run {sex.run_ts} · "
           f"commit {sex.git_commit} · Data: NHTSA FARS (US public domain).")
st.caption("[Impressum](/Impressum) · [Datenschutz](/Datenschutz)")
