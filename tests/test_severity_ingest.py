"""Tests for the v2 severity normalizers, frontal classifiers, cohort SQL and
the analyze_v2 estimand contract - all on planted frames/rows, no downloads."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from crashgap.db import insert_frame
from crashgap.ingest_severity import (
    SEV_COLUMNS,
    belted_flag,
    ciss_frontal,
    nass_frontal,
    sex_female,
    valid_mais,
)
from crashgap.severity import analyze_v2, severity_cohort, source_years


def test_sex_female_includes_pregnancy_codes():
    s = pd.Series([1, 2, 3, 4, 5, 6, 9, np.nan])
    out = sex_female(s)
    assert list(out[:7]) == [0, 1, 1, 1, 1, 1, pd.NA][:7] or (
        out.iloc[0] == 0 and all(out.iloc[1:6] == 1) and pd.isna(out.iloc[6]))
    assert pd.isna(out.iloc[7])


def test_belted_flag_excludes_type_unknown_and_child_combos():
    s = pd.Series([2, 3, 4, 5, 0, 1, 14, 99, np.nan])
    out = belted_flag(s)
    assert all(out.iloc[:3] == 1)
    assert out.iloc[4] == 0  # affirmative none
    for i in (3, 5, 6, 7, 8):  # type-unknown, inoperative, child combo, unknown, missing
        assert pd.isna(out.iloc[i])


def test_valid_mais_nulls_unknown_codes():
    s = pd.Series([0, 3, 6, 7, 9, 99, np.nan])
    out = valid_mais(s)
    assert list(out.iloc[:3]) == [0, 3, 6]
    assert all(pd.isna(v) for v in out.iloc[3:])


def test_ciss_frontal_uses_primary_event():
    cdc = pd.DataFrame({
        "CASENUMBER": [1, 1, 2, 2, 3],
        "VEHNO": [1, 1, 1, 1, 1],
        "EVENTNO": [1, 2, 1, 2, 1],
        "DVRANK": [2, 1, 8, 8, 1],
        "CDCPLANE": ["L", "F", "F", "L", "F"],
        "OCLOCK": [9, 12, 12, 9, 6],
    })
    out = ciss_frontal(cdc).set_index(["CASENUMBER", "VEHNO"])["is_frontal"]
    assert out.loc[(1, 1)] == 1  # rank-1 event is frontal, rank-2 side ignored
    assert out.loc[(2, 1)] == 1  # no ranked event: lowest EVENTNO wins, it's frontal
    assert out.loc[(3, 1)] == 0  # plane F but 6 o'clock is not frontal


def test_nass_frontal_requires_gad_f_and_clock():
    ve = pd.DataFrame({
        "PSU": [1, 1, 1, 1],
        "CASENO": [1, 2, 3, 4],
        "VEHNO": [1, 1, 1, 1],
        "GAD1": [b"F", b"F", b"L", "F"],  # bytes and str both occur
        "DOF1": [12, 6, 12, 1],
    })
    out = nass_frontal(ve).set_index("CASENO")["is_frontal"]
    assert list(out.loc[[1, 2, 3, 4]]) == [1, 0, 0, 1]


def _plant_sev_rows(conn, n: int = 400, source: str = "ciss",
                    year: int = 2020) -> None:
    rng = np.random.default_rng(20260901)
    rows = []
    for i in range(n):
        female = int(i % 2)
        age = int(rng.integers(20, 80))
        mais = int(rng.integers(0, 6)) if rng.random() > 0.05 else None
        dv = float(rng.uniform(10, 70)) if rng.random() < 0.6 else None
        height = float(rng.uniform(150, 200)) if rng.random() < 0.7 else None
        bmi = float(rng.uniform(18, 40)) if height is not None else None
        rows.append((
            source, year, int(rng.integers(1, 9)), int(rng.integers(1, 4)),
            str(i), 1, 1, float(rng.uniform(10, 500)), female, age,
            int(rng.choice([11, 13])), 1, height, None, bmi, mais, None,
            4, int(rng.choice([1995, 2005, 2012, 2020])), dv, 1,
        ))
    # plus rows every cohort filter must drop
    rows += [
        (source, year, 1, 1, "u1", 1, 1, 100.0, None, 40, 11, 1, None, None,
         None, 2, None, 4, 2012, None, 1),      # unknown sex
        (source, year, 1, 1, "u2", 1, 1, 100.0, 1, 40, 21, 1, None, None,
         None, 2, None, 4, 2012, None, 1),      # rear seat
        (source, year, 1, 1, "u3", 1, 1, 100.0, 1, 40, 11, 0, None, None,
         None, 2, None, 4, 2012, None, 1),      # unbelted
        (source, year, 1, 1, "u4", 1, 1, 100.0, 1, 40, 11, 1, None, None,
         None, 2, None, 4, 2012, None, 0),      # not frontal
        (source, year, 1, 1, "u5", 1, 1, 100.0, 1, 12, 11, 1, None, None,
         None, 2, None, 4, 2012, None, 1),      # under the age window
        (source, year, 1, 1, "u6", 1, 1, 100.0, 1, 40, 11, 1, None, None,
         None, None, None, 4, 2012, None, 1),   # ungraded MAIS
        (source, year, 1, 1, "u7", 1, 1, 100.0, 1, 40, 11, 1, None, None,
         None, 2, None, 80, 2012, None, 1),     # not a light vehicle
    ]
    insert_frame(conn, "sev_occupant", SEV_COLUMNS, rows)


def test_severity_cohort_applies_every_filter(conn):
    _plant_sev_rows(conn, n=100)
    cohort = severity_cohort(conn, "ciss")
    dropped = {"u1", "u2", "u3", "u4", "u5", "u6", "u7"}
    assert dropped.isdisjoint(set(cohort["case_id"]))
    # only the planted eligible rows with a graded MAIS survive
    assert len(cohort) == 100 - int(
        conn.execute("SELECT COUNT(*) FROM sev_occupant WHERE mais IS NULL "
                     "AND case_id NOT LIKE 'u%'").fetchone()[0])
    assert source_years(conn, "ciss") == "2020-2020"
    assert source_years(conn, "nass") is None


def test_analyze_v2_writes_the_full_estimand_contract(conn):
    _plant_sev_rows(conn, n=400, source="ciss", year=2020)
    _plant_sev_rows(conn, n=400, source="nass", year=2010)
    conn.execute("UPDATE sev_occupant SET mais08 = mais WHERE source = 'nass'")
    conn.commit()

    run_ts = analyze_v2(conn)
    rows = pd.read_sql_query(
        "SELECT estimand, fars_years, cohort_def, point FROM results WHERE run_ts = ?",
        conn, params=(run_ts,))

    expected = set()
    for source in ("ciss", "nass"):
        for outcome in ("mais2plus", "mais3plus"):
            for tier in ("base", "dv", "dvanthro"):
                expected.add(f"{source}_svylogit_female_or_{outcome}_frontal_{tier}")
            expected.add(f"{source}_svylogit_female_or_{outcome}_frontal_base_wtrim95")
            if source == "nass":
                expected.add(f"{source}_svylogit_female_or_{outcome}_frontal_base_ais08")
                expected.add(f"{source}_svylogit_female_or_{outcome}_frontal_base_ais90dual")
    assert set(rows["estimand"]) == expected
    assert len(rows) == len(expected) == 20

    years = dict(zip(rows["estimand"], rows["fars_years"]))
    assert years["ciss_svylogit_female_or_mais2plus_frontal_base"] == "2020-2020"
    assert years["nass_svylogit_female_or_mais2plus_frontal_base"] == "2010-2010"

    for _, row in rows.iterrows():
        info = json.loads(row["cohort_def"])
        for key in ("n_obs", "n_events", "n_psus", "n_strata", "df",
                    "kish_neff", "weight_max_over_median", "design"):
            assert key in info, f"{key} missing in {row['estimand']}"
        if info["n_events"] >= 30 and info["df"] >= 1:
            assert row["point"] is not None

    # planted data has NO female effect: the base ORs must hug 1
    base = rows[rows.estimand == "ciss_svylogit_female_or_mais2plus_frontal_base"]
    assert 0.6 < float(base["point"].iloc[0]) < 1.6


def test_analyze_v2_skips_absent_sources(conn):
    _plant_sev_rows(conn, n=60, source="ciss", year=2020)
    run_ts = analyze_v2(conn)
    rows = pd.read_sql_query(
        "SELECT estimand FROM results WHERE run_ts = ?", conn, params=(run_ts,))
    assert all(e.startswith("ciss_") for e in rows["estimand"])


def test_ais08_rows_use_the_dual_coded_outcome(conn):
    _plant_sev_rows(conn, n=300, source="nass", year=2012)
    # dual-code half the graded rows one grade HIGHER than MAIS
    conn.execute("UPDATE sev_occupant SET mais08 = MIN(mais + 1, 6) "
                 "WHERE source='nass' AND mais IS NOT NULL AND (CAST(case_id AS INTEGER)) % 2 = 0")
    conn.commit()
    run_ts = analyze_v2(conn)
    rows = pd.read_sql_query(
        "SELECT estimand, cohort_def FROM results WHERE run_ts = ?",
        conn, params=(run_ts,))
    ais08 = rows[rows.estimand == "nass_svylogit_female_or_mais2plus_frontal_base_ais08"]
    info = json.loads(ais08["cohort_def"].iloc[0])
    assert info["ais_revision"] == "AIS2008"
    assert info["dual_coded_subset"] is True
    # the ais08 fit runs on the dual-coded subset only
    n_dual = conn.execute(
        "SELECT COUNT(*) FROM sev_occupant WHERE source='nass' AND mais08 IS NOT NULL "
        "AND mais IS NOT NULL AND sex_female IS NOT NULL AND belted=1 AND is_frontal=1 "
        "AND seat_pos IN (11,13) AND age BETWEEN 16 AND 96 "
        "AND case_id NOT LIKE 'u%'").fetchone()[0]
    assert info["n_obs"] == n_dual

    # the ais90dual twin: SAME subset, graded by AIS90 mais, so the pair
    # isolates the revision; its ledger row carries the SUBSET's year span
    dual90 = rows[rows.estimand == "nass_svylogit_female_or_mais2plus_frontal_base_ais90dual"]
    info90 = json.loads(dual90["cohort_def"].iloc[0])
    assert info90["ais_revision"] == "AIS90"
    assert info90["dual_coded_subset"] is True
    assert info90["n_obs"] == n_dual
    years = pd.read_sql_query(
        "SELECT estimand, fars_years FROM results WHERE run_ts = ?", conn,
        params=(run_ts,))
    span = dict(zip(years["estimand"], years["fars_years"]))
    assert span["nass_svylogit_female_or_mais2plus_frontal_base_ais90dual"] == "2012-2012"
    assert span["nass_svylogit_female_or_mais2plus_frontal_base_ais08"] == "2012-2012"
