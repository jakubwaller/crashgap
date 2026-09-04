"""CrashGap dashboard.

Leads with the seat/sex split, never with the pooled ratio. The pooled
double-pair number is seat-confounded (seat position is the one attribute
that differs within every pair and it is ~73/27 collinear with sex), so on
its own it is the number a skeptic dismantles in one query.

Two windows, kept apart: the headline reads the modern decade (the newest
MODERN_WINDOW calendar years ingested), so "does being female raise fatality
risk in the same crash today" never quietly absorbs 1990s vehicles as more
history is ingested; the trend and same-sex sections read the full ingested
span, where the older model-year bands have power. Every number says which
window it comes from.

Each section is a chart or tile plus a few plain sentences. The full prose
(caveats, specification checks, method) sits in expanders on the same page,
never in a tooltip.
"""

from __future__ import annotations

import json
import math
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
        {"configuration": "she is the passenger", "ratio": core.rr_she_passenger,
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
st.caption("Do women die more often than men in the same crash? Public US crash data (NHTSA), "
           "recomputed when a new year comes out.")
st.caption("The idea came from reading [*Invisible Women*](https://en.wikipedia.org/wiki/Invisible_Women:_Exposing_Data_Bias_in_a_World_Designed_for_Men) "
           "by Caroline Criado Perez.")

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

modern_rows = results.set_index("estimand")
sex = modern_rows.loc[SEX + "core"]
seat = modern_rows.loc[SEAT + "core"]
pooled = modern_rows.loc[POOLED + "core"]
strata = load_strata(DB_PATH, mspan)
core, headon = strata["core"], strata["headon"]
sev = load_severity(DB_PATH)

CUT_LABEL = {"core": "frontal, 11 to 1 o'clock (the headline)",
             "wide": "frontal, 10 to 2 o'clock",
             "headon": "head-on only"}


def clear(ci_lo: float, ci_hi: float, null: float = 1.0) -> bool:
    """Whether a 95% interval excludes its null."""
    return ci_lo > null or ci_hi < null


# The short version: three sentences a first-time reader can stop after. The
# female effect is a range because the frontal cuts disagree and quoting
# either one alone is cherry-picking (see the first expander).
_effects = [((d.sex_effect - 1) * 100, d) for d in strata.values() if d.sex_effect is not None]
if _effects:
    (lo, _lo_cut), (hi, _) = min(_effects, key=lambda e: e[0]), max(_effects, key=lambda e: e[0])
    # one phrase for the range, reused below; a low end at or below zero is said as such
    range_phrase = (f"between about zero and {hi:.0f}%" if lo <= 0
                    else f"between roughly {lo:.0f}% and {hi:.0f}%")
    low_end_zero = lo > 0 and not _lo_cut.sex_is_significant
else:
    lo = hi = range_phrase = None
    low_end_zero = False
share = 100 * core.n_pairs_she_passenger / core.n_pairs
headon_flip = headon.seat_effect is not None and headon.seat_effect < 1 \
    and headon.seat_is_significant
if range_phrase is None:
    female_range = "being a woman adds an amount this run could not estimate"
else:
    female_range = (f"being a woman adds {range_phrase} to the risk of being the one who died, "
                    f"depending on the type of frontal crash"
                    + (", and the low end could be zero" if low_end_zero else ""))
st.markdown(f"Averaged over fatal frontal crashes, the passenger seat is the more dangerous seat"
            f"{' (head-on collisions are the exception)' if headon_flip else ''}. Women sit "
            f"there more often, and that explains most of the raw gap. Once the seat is "
            f"accounted for, {female_range}. Cars built before 2000 were clearly worse for "
            f"women than 2000s cars, and newer cars show no further improvement. Injury data, "
            f"which measures something different, shows a bigger gap.")

st.subheader("Who died, her or him?")
left, right = st.columns(2)
left.metric("Being the woman, same seat", pct(sex.point),
            delta=f"95% CI {pct(sex.ci_lo)} to {pct(sex.ci_hi)}", delta_color="off")
right.metric("Sitting in the passenger seat", pct(seat.point),
             delta=f"95% CI {pct(seat.ci_lo)} to {pct(seat.ci_hi)}", delta_color="off")
st.markdown(f"{int(sex.n_pairs):,} pairs, a man and a woman belted in the front seats of the "
            f"same car, in {int(sex.n_crashes):,} fatal frontal crashes (FARS {sex.fars_years}). "
            f"The seat effect {'is clear' if clear(seat.ci_lo, seat.ci_hi) else 'could be zero'}. "
            f"The female effect {'is clear' if clear(sex.ci_lo, sex.ci_hi) else 'could be zero'}.")
st.warning(f"Without the seat split, women come out **{pooled.point:.2f}×** as likely to be the "
           f"one who died. Most of that is the seat: women were the passenger in "
           f"{share:.0f}% of these pairs.")

st.markdown(f"**She died vs he died, by who drove (FARS {MODERN})**")
st.altair_chart(split_chart(core), width="stretch")
st.caption("Above 1.0 she died more often. The drop below 1.0 when she drives is the seat "
           "effect.")

with st.expander("The counts, and what changes with the crash type"):
    st.dataframe(pd.DataFrame([
        {"who sat where": "she is the passenger, he drives",
         "pairs": core.n_pairs_she_passenger,
         "she died : he died": core.rr_she_passenger},
        {"who sat where": "she drives, he is the passenger",
         "pairs": core.n_pairs_she_drives,
         "she died : he died": core.rr_she_drives},
    ]), hide_index=True)
    st.caption("If only sex mattered, the two ratios would agree. The square root of their "
               "product is the sex effect and the square root of their quotient is the seat "
               "effect (Evans' double-pair method).")
    st.dataframe(pd.DataFrame([
        {"crash type": CUT_LABEL[v],
         "sex effect": pct(d.sex_effect),
         "sex 95% CI": f"{pct(d.sex_ci[0])} to {pct(d.sex_ci[1])}",
         "sex clear": "yes" if d.sex_is_significant else "no",
         "seat effect": pct(d.seat_effect),
         "seat clear": "yes" if d.seat_is_significant else "no",
         "pairs": d.n_pairs}
        for v, d in strata.items()
    ]), hide_index=True)
    _flip = (" and the driver's seat is the worse one, plausibly because head-on collisions "
             "mostly hit the driver's side (the driver-side small-overlap crash test existed "
             "six years before a passenger-side one)") if headon_flip else ""
    _range = (f" So the female effect sits {range_phrase} depending on crash geometry, and this "
              f"data cannot pin it down further.") if range_phrase is not None else ""
    st.warning(f"Head-on crashes differ. There the female effect is **{pct(headon.sex_effect)}** "
               f"({'clear' if headon.sex_is_significant else 'not clear'}){_flip}.{_range} "
               f"Quoting only one row would be misleading.")

st.subheader("Does the gap shrink as cars get newer?")
condlogit_bands = load_condlogit_bands(DB_PATH, FULL, span)
st.markdown(f"Partly. Cars built before 2000 were clearly worse for women than 2000s cars. After "
            f"that, nothing improves further in this data (FARS {FULL}, adjusted for age).")
_tchart = trend_chart(condlogit_bands)
if _tchart is not None:
    st.altair_chart(_tchart, width="stretch")
    st.caption(f"How much more often the woman died, by the car's model year, after accounting "
               f"for age, with a 95% range. Above 1.0 the woman died more often. Grey marks "
               f"rest on fewer than {POWER_FLOOR} informative pairs and are directional only.")

with st.expander("The trend in detail"):
    st.markdown(f"""
The published finding (Atwood, Noh & Craig 2023) is that the female penalty fell from 19.9% in the
oldest cars to 5.8% in the newest. The test here is a within-car conditional logit over FARS
{FULL} with seat and age held constant and one female coefficient per model-year band, plus two
refits: seat and age fitted separately inside each band, and only cars at most {VEHICLE_AGE_MAX}
years old at the crash.

Before 2000 the penalty is about +23% to +28%, in the 2000s about +5% to +11%, with
non-overlapping intervals under all three specifications. That reproduces the direction of the
published decline. After the 2000s nothing declines further. The 2010–2016 band sits above the
2000s level in all three specifications, and the newest band does too except under the per-band
refit, where the two are indistinguishable (1.10 vs 1.11). The default fit even puts the newest
band significantly above the 2000s (Wald contrast, p ≈ 0.03), but that does not survive the
per-band refit, and no post-2000 contrast has non-overlapping intervals under any specification.
It is a model-dependent hint and is not counted as a finding.

The newest band moves with that choice (1.26 pooled vs 1.10 per band) because the seat effect
differs by band, about 1.0 in older cars and 1.34 [1.17–1.53] in 2017+, so the data reject
pooling exactly where it matters. The continuous slope is −0.025 log-odds per decade (95% CI
−0.062 to +0.013), consistent with a slow decline and with zero. The published decline into the
newest cars may be hidden by the 2020–2024 years this window adds beyond that study's 2019
endpoint, or it may be a real plateau. This design cannot tell.
""")

    def _grey_low_power(r: pd.Series) -> list[str]:
        if r["headline safe"] == "low power":
            return ["color: #888888; background-color: rgba(128,128,128,0.12)"] * len(r)
        return [""] * len(r)

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
    st.caption(f"Grey rows have fewer than {POWER_FLOOR} discordant pairs and are directional "
               "only. The naive columns use the pooled, seat-confounded ratio without age "
               "adjustment; women in these pairs average 1.4 years younger, so those columns "
               "understate the female OR.")

    specs = load_band_specs(DB_PATH, FULL)
    if not specs.empty:
        st.markdown("The same per-band female OR under three specifications. A band claim that "
                    "does not hold in all three columns depends on a modelling choice.")
        st.dataframe(specs.rename(columns={"mod_year_band": "vehicle model years"}),
                     hide_index=True)
        st.caption(f"*Separate nuisance* fits seat and age inside each band instead of pooling "
                   f"them; the data reject pooling, since the per-band seat effect has "
                   f"non-overlapping intervals. *Vehicle age ≤ {VEHICLE_AGE_MAX}* keeps only "
                   f"the first {VEHICLE_AGE_MAX} years of each car's life, which makes band and "
                   f"calendar period nearly the same thing, as the published FARS trend "
                   f"estimates do.")

    trend = load_condlogit_trend(DB_PATH, FULL)
    if trend is not None and pd.notna(trend.point):
        st.caption(f"Continuous check, fitted separately from the bands: "
                   f"**{trend.point:+.4f}** log-odds per decade (95% CI {trend.ci_lo:+.4f} "
                   f"to {trend.ci_hi:+.4f}). It is not a third headline number.")

st.subheader("Is the passenger seat worse for women specifically?")
st.markdown(f"A man and a woman in the same car cannot tell whether the seat effect differs by "
            f"sex, because in every such pair she is the woman. So cars with two men are "
            f"compared with cars with two women (FARS {FULL}).")
samesex = load_samesex_interaction(DB_PATH, FULL)
male_seat, female_seat = samesex["male"], samesex["female"]
male_adj, female_adj = samesex["male_ageadj"], samesex["female_ageadj"]
interaction = samesex["interaction"]
interaction_adj = samesex["interaction_ageadj"]
interaction_cmp = samesex["interaction_agecomparable"]


def _row_headline_safe(r: pd.Series | None) -> bool:
    """A results row counts as renderable only if it exists, its point isn't
    the NaN a NULL SQL REAL round-trips to (a plain `is not None` check does
    not catch this - `r` itself is a real Series, not None, even when
    every value in it is NaN), and it clears the same power floor the band
    table above enforces. This is the most power-starved section on the
    page (design doc section 6), so it is exactly where a missing/silent
    floor check would first bite."""
    if r is None or pd.isna(r.point):
        return False
    n = _discordant_count(r.cohort_def)
    return n is None or n >= POWER_FLOOR


male_ok, female_ok = _row_headline_safe(male_adj), _row_headline_safe(female_adj)
interaction_ok = _row_headline_safe(interaction_adj)
raw_ok = all(_row_headline_safe(r) for r in (male_seat, female_seat, interaction))

# The directional claim renders only when the estimate backing it exists,
# clears the power floor, and is itself significant; the design doc requires
# the null reading and the weaker-assumption note next to the number, not
# below the fold (docs/v1-design.md sections 3 and 6). The claim is about the
# difference between the two seat effects, so it stays true whichever way the
# individual baselines point.
WEAKER = "This rests on a weaker design than the rest of the page."
if male_ok and female_ok and interaction_ok:
    if interaction_adj.ci_lo > 0:
        st.markdown(f"Adjusted for age, the passenger seat is worse for a woman riding with a "
                    f"woman than for a man riding with a man. How much worse depends on the age "
                    f"model. {WEAKER}")
    elif interaction_adj.ci_hi < 0:
        st.markdown(f"Adjusted for age, the passenger seat is less bad for a woman riding with a "
                    f"woman than for a man riding with a man. How much depends on the age "
                    f"model. {WEAKER}")
    else:
        st.markdown(f"Adjusted for age, the difference between the two could be zero in this "
                    f"run. At this sample size that is not strong evidence of no effect. "
                    f"{WEAKER}")

sx_left, sx_right = st.columns(2)
if male_ok:
    sx_left.metric("Two men: passenger vs driver, age-adjusted",
                   f"{male_adj.point:.3f}×",
                   delta=f"95% CI {male_adj.ci_lo:.3f} to {male_adj.ci_hi:.3f}",
                   delta_color="off")
else:
    sx_left.info("Two-men seat baseline not available or below the power floor in this run.")
if female_ok:
    sx_right.metric("Two women: passenger vs driver, age-adjusted",
                    f"{female_adj.point:.3f}×",
                    delta=f"95% CI {female_adj.ci_lo:.3f} to {female_adj.ci_hi:.3f}",
                    delta_color="off")
else:
    sx_right.info("Two-women seat baseline not available or below the power floor in this run.")

if male_ok and female_ok and interaction_ok:
    ix_left, ix_right = st.columns(2)
    ix_left.metric("Difference between the two (log-ratio)",
                   f"{interaction_adj.point:+.3f}",
                   delta=f"95% CI {interaction_adj.ci_lo:+.3f} to {interaction_adj.ci_hi:+.3f}",
                   delta_color="off")
    if _row_headline_safe(interaction_cmp):
        ix_right.metric("Same difference, pairs within 10 years of age",
                        f"{interaction_cmp.point:+.3f}",
                        delta=f"95% CI {interaction_cmp.ci_lo:+.3f} to "
                              f"{interaction_cmp.ci_hi:+.3f}",
                        delta_color="off")
        st.caption(f"Above 1.0× the passenger dies more often. In percentages, the passenger-seat "
                   f"penalty is about {(math.exp(interaction_adj.point) - 1) * 100:+.0f}% "
                   f"for two women compared with two men, and about "
                   f"{(math.exp(interaction_cmp.point) - 1) * 100:+.0f}% when only pairs within "
                   f"10 years of age are compared. Read the two together. The second number "
                   f"shows how much of the first rests on pairs with a large age gap, where the "
                   f"age model is least reliable.")
    with st.expander("Why two cars, the raw numbers, and the weaker assumption"):
        st.markdown("""
In a mixed pair the sex contrast is the same every time, so a female × seat term is just the
seat effect again. The way around it is to compute passenger-vs-driver fatality odds
inside two-men cars and inside two-women cars and compare them.

The raw numbers are confounded by age. In a two-women car the passenger is much more often far
older than the driver, and age predicts who dies more strongly than seat or sex. The headline
numbers are therefore age-adjusted (a quadratic in the within-pair age difference, the same
machinery as the mixed-sex model).
""")
        if raw_ok:
            st.caption(f"Unadjusted: two men {male_seat.point:.3f}× "
                       f"[{male_seat.ci_lo:.3f}, {male_seat.ci_hi:.3f}], two women "
                       f"{female_seat.point:.3f}× [{female_seat.ci_lo:.3f}, "
                       f"{female_seat.ci_hi:.3f}], raw log-ratio {interaction.point:+.3f} "
                       f"[{interaction.ci_lo:+.3f}, {interaction.ci_hi:+.3f}]. The gap to the "
                       f"age-adjusted numbers is the age confound. Do not quote the raw "
                       f"asymmetry as a seat or sex effect.")
        st.caption("The age-adjusted difference is positive and significant, and its direction "
                   "holds across calendar eras. Its size depends on the age model: within pairs of "
                   "similar age it roughly halves and its interval touches zero. Together the "
                   "two numbers support a female-specific passenger penalty of roughly +10% to "
                   "+23%, carried mostly by pairs with a large age gap.")
        st.caption("The weaker assumption: two men or two women are not paired with the other "
                   "sex in the same crash, so who rides with whom could correlate with crash "
                   "severity or vehicle type. This is also the section with the least data, "
                   "because two-occupant fatal crashes are mostly mixed-sex.")
else:
    st.info(f"Same-sex comparison not available, or below the {POWER_FLOOR}-discordant-pair "
            f"power floor, in this run.")

if not sev.empty:
    ciss_years = sev[sev.estimand.str.startswith("ciss_")]["fars_years"].iloc[0] \
        if (sev.estimand.str.startswith("ciss_")).any() else None
    nass_years = sev[sev.estimand.str.startswith("nass_")]["fars_years"].iloc[0] \
        if (sev.estimand.str.startswith("nass_")).any() else None
    st.subheader("The injury gap")
    st.markdown(f"FARS only knows who died. CISS ({ciss_years or '—'}) and NASS-CDS "
                f"({nass_years or '—'}) grade injuries and reconstruct crash forces, which is "
                f"where the published numbers come from. Where a same-era published number "
                f"exists, the delta-V-adjusted estimate here lands close to it without tuning.")
    _schart = severity_chart(sev)
    if _schart is not None:
        st.altair_chart(_schart, width="stretch")
        st.caption("How much more likely women were to be moderately (MAIS 2+) or seriously "
                   "(MAIS 3+) hurt, belted front-seat adults in frontal crashes, after "
                   "accounting for how hard the crash hit, with a 95% range. Above 1.0 women "
                   "get hurt more. Faded marks rest on too few cases. Diamonds are the "
                   "published numbers, each in its own era's data.")
    sev_expander = st.expander("The benchmark table, the weights and the injury coding")
    sev_expander.markdown("""
CISS and its predecessor NASS-CDS are weighted national samples of tow-away crashes with
hospital-grade injury coding (AIS) and reconstructed delta-V. The estimate is the female odds
ratio for **MAIS 2+** (Craig 2024's quantity, published ~1.75) and **MAIS 3+** (Bose 2011's,
published ~1.47), from a design-weighted logistic regression with survey-style standard errors
and t-based intervals on the design's own degrees of freedom (15 to 28 here, so the intervals
are wide). Three covariate tiers: **base** (age, seat, model-year band), **+ΔV** (adds delta-V
and its square, where reconstruction succeeded) and **+ΔV+anthro** (adds height and BMI, which
partly carry the sex effect, so that tier says what remains net of body size rather than whether
the gap is real).
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
    sev_expander.caption("The benchmark column repeats each published number in both data sets. "
                         "Only the era-matched cells are like for like: CISS MAIS 2+ against "
                         "Craig 2024 and NASS MAIS 3+ against Bose 2011. The other two cells "
                         "have no published number from their own era.")

    ciss_base = _sev_row(sev, "ciss_svylogit_female_or_mais2plus_frontal_base")
    nass_base = _sev_row(sev, "nass_svylogit_female_or_mais3plus_frontal_base")
    ais08 = _sev_row(sev, "nass_svylogit_female_or_mais3plus_frontal_base_ais08")
    ais90dual = _sev_row(sev, "nass_svylogit_female_or_mais3plus_frontal_base_ais90dual")
    diag_bits = []
    for label, r in (("CISS", ciss_base), ("NASS", nass_base)):
        if r is not None and pd.notna(r.point):
            diag_bits.append(
                f"{label}: n={int(r.n_obs):,}, events={int(r.n_events):,}, "
                f"design df={int(r.df)}, Kish effective n={r.kish_neff:,.0f}, "
                f"max/median weight={r.weight_max_over_median:.0f}×")
    if diag_bits:
        sev_expander.caption("Weights: " + "; ".join(diag_bits) + ". A max/median weight in the "
                             "hundreds means single cases can move an estimate (the Viano 2025 "
                             "critique), hence the trimmed column and the design-df intervals.")
    if ais08 is not None and pd.notna(ais08.point):
        pair = (f"the same {ais08.fars_years} dual-coded NASS cohort grades MAIS 3+ at "
                f"{_sev_fmt(ais90dual)} under AIS90 and {_sev_fmt(ais08)} under AIS2008"
                if ais90dual is not None and pd.notna(ais90dual.point)
                else f"the {ais08.fars_years} dual-coded NASS cohort grades MAIS 3+ at "
                     f"{_sev_fmt(ais08)} under AIS2008")
        sev_expander.caption(f"Injury coding: {pair}. Same cohort, so that difference is the "
                             f"coding itself, and part of any Bose-vs-Craig gap is coding rather "
                             f"than crash physics.")
    sev_expander.markdown("""
Delta-V adjustment raises every estimate, because women's frontal crashes carry lower delta-V on
average, the direction every published severity analysis reports. Height and BMI do not absorb
the gap: they leave NASS MAIS 3+ unchanged, raise both MAIS 2+ cells and lower only the CISS
MAIS 3+ cell, which is not significant in either tier. Trimming the weights shifts the point estimates
visibly and leaves every direction unchanged. The table supports a female odds ratio in frontal crashes roughly in the 1.4 to
2.0 band the literature reports, with the delta-V-adjusted estimates in its upper half.

Only occupants with a completed injury workup enter, as in every published analysis of these
files. Delta-V is missing more often in the worst crashes, so base vs +ΔV is partly a change of
cohort. This is not a reproduction of Bose or Craig: frontal definitions, cohorts, AIS revisions
and covariates differ (docs/v2-design.md). Nothing here is pooled with FARS.
""")

caveats = st.expander("Before you quote anything from this page")
caveats.markdown("""
- Every FARS number here is a relative risk within a fatal frontal crash. None of it is a
  population rate.
- The shared car controls the crash but not age. Women in these pairs average 1.3 to 1.4 years
  younger. The `fars_doublepair_*` rows do not adjust for that; the `fars_condlogit_*` and
  `_ageadj` rows do.
- The published severity-adjusted gap (Bose 2011, MAIS 3+ odds ~1.47) is a different, larger
  quantity, measured in the injury section. The FARS tiles do not refute it.
- Belt use and injury severity in FARS are police codes, nobody measured them.
- Two FARS recodes sit inside the full window (unknown-age code 2009, impact-point codes 2010).
  Both are harmless for the core cohort (`codebook.py`). The *wide* frontal variant selects a
  somewhat narrower crash set after 2010, so cross-era comparisons of that one variant carry an
  asterisk.
- The conditional logit is a standard matched-pairs model. It has not been checked against the
  exact equations of the published trend studies.
- The default trend model pools seat and age across model-year bands, which the data reject.
  Only band contrasts that also hold under the separate-nuisance and vehicle-age-capped refits
  count.
- Model year, vehicle age and calendar period cannot all be separated, and none varies within a
  car. Where the specifications disagree, the claim is not identified.
- Age enters as a quadratic, which will miss a sharp threshold past roughly 65 to 75. A
  piecewise-linear variant (knot at 65) is run alongside it, and {piecewise_agreement} Neither
  form is verified against the true fatality-age curve.
- The seat × sex question rests entirely on the two-men and two-women cars and their weaker
  assumption. A null result there is weak evidence. Quote only the `_ageadj` rows, together with
  `_agecomparable`.
- No covariate reaches crash severity directly. A shift in the mix of survivable fatal crashes
  across model-year bands would survive the age adjustment.
- A number backed by fewer than {floor} discordant pairs is written to the results table and
  never headlined; read it as directional only.
""".format(piecewise_agreement=piecewise_agreement_text(condlogit_bands, FULL),
           floor=POWER_FLOOR))

method = st.expander("How the numbers are computed")
_adj_note = ""
_name = CONDLOGIT_SEX_BAND + "core_pooled"
if _name in modern_rows.index and pd.notna(modern_rows.loc[_name].point):
    _p = modern_rows.loc[_name]
    _adj_note += (f" Its pooled female effect ({_p.point:.2f}× on FARS {MODERN}) is not a "
                  f"replacement for the seat-balanced tile above: one nets out the seat by "
                  f"geometric means without age, the other pools an age-adjusted band model "
                  f"with one shared seat term.")
_name = "fars_condlogit_seat_effect_rightfront_frontal_core"
if _name in modern_rows.index and pd.notna(modern_rows.loc[_name].point):
    _s = modern_rows.loc[_name]
    _adj_note += (f" Its seat effect ({_s.point:.2f}×, "
                  f"{'clear' if clear(_s.ci_lo, _s.ci_hi) else 'could be zero'}) is "
                  + ("smaller than the seat tile, because age adjustment and the band-varying "
                     "seat effect absorb part of it." if _s.point < seat.point
                     else "close to the seat tile."))
method.markdown(f"""
Evans' double-pair design. Only cars carrying exactly one eligible man and one eligible woman
enter (both belted, front outboard, age 16 to 96, light vehicle), so the crash, the car and the
impact are the same for both. The estimate is the ratio of pairs where she died and he did not
to pairs where he died and she did not. Run separately by seat configuration, the two ratios
measure *f·s* and *f/s* (*f* the female multiplier, *s* the passenger-seat multiplier), so the
square root of their product is *f* and the square root of their quotient is *s*.

Frontal means IMPACT1 in {{11, 12, 1}} (Forman 2019); the wider fan and the head-on cut are in
the first expander. Intervals are percentile bootstraps over whole crashes, both seat strata
resampled inside one replicate.

The headline uses FARS {MODERN}, the most recent decade. The trend and same-sex sections use
the full span, FARS {FULL}, where the older model-year bands have enough pairs. Every FARS code the cohort depends
on was re-checked across the 2000-vintage files (`codebook.py`).

The age-adjusted numbers come from a within-car conditional logit: a logistic regression of
"the woman died" on her covariates minus his (seat, quadratic age difference) over discordant
pairs, with standard errors clustered on the crash.{_adj_note}

The finding itself is settled science (Evans 1988; Bose 2011; Forman 2019; Atwood, Noh & Craig
2023). This site adds the live, versioned form.
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
    st.caption("`n_pairs` is the eligible cohort including concordant pairs. "
               "`n_discordant_pairs` is what the fit used (blank for the double-pair rows) and "
               "is the one that says how much power a row has.")

st.caption(f"Headline: FARS {MODERN} · trend/same-sex: FARS {FULL} · run {sex.run_ts} · "
           f"commit {sex.git_commit} · Data: NHTSA FARS (US public domain).")
st.caption("[Impressum](/Impressum) · [Datenschutz](/Datenschutz)")
