"""SQLite storage: FARS (and later CISS) side by side, keyed by (source, year).

Nullable mais/weight stay NULL for FARS and get populated when the CISS rung
lands, so v0 -> v2 needs no migration. Inserts are INSERT OR REPLACE on the
natural key, which keeps re-ingests of the same year idempotent.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from . import codebook as cb

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS accident (
    source TEXT NOT NULL, year INTEGER NOT NULL, st_case INTEGER NOT NULL,
    state INTEGER, man_coll INTEGER, harm_ev INTEGER,
    ve_total INTEGER, persons INTEGER, fatals INTEGER,
    PRIMARY KEY (source, year, st_case)
);
CREATE TABLE IF NOT EXISTS vehicle (
    source TEXT NOT NULL, year INTEGER NOT NULL, st_case INTEGER NOT NULL,
    veh_no INTEGER NOT NULL,
    body_typ INTEGER, mod_year INTEGER, impact1 INTEGER,
    deformed INTEGER, numoccs INTEGER, rollover INTEGER,
    PRIMARY KEY (source, year, st_case, veh_no)
);
CREATE TABLE IF NOT EXISTS person (
    source TEXT NOT NULL, year INTEGER NOT NULL, st_case INTEGER NOT NULL,
    veh_no INTEGER NOT NULL, per_no INTEGER NOT NULL,
    per_typ INTEGER, sex INTEGER, age INTEGER, seat_pos INTEGER,
    rest_use INTEGER, air_bag INTEGER, inj_sev INTEGER,
    mais INTEGER,   -- CISS/NASS-CDS only, else NULL
    weight REAL,    -- survey weight; CISS/NASS-CDS only, else NULL
    PRIMARY KEY (source, year, st_case, veh_no, per_no)
);
CREATE INDEX IF NOT EXISTS idx_person_key  ON person  (source, year, st_case, veh_no);
CREATE INDEX IF NOT EXISTS idx_vehicle_key ON vehicle (source, year, st_case, veh_no);

-- results: the live, versioned, queryable citation layer
CREATE TABLE IF NOT EXISTS results (
    run_ts TEXT NOT NULL, git_commit TEXT NOT NULL,
    estimand TEXT NOT NULL,      -- e.g. 'fars_doublepair_fatality_f_vs_m_frontal_core'
    fars_years TEXT NOT NULL,    -- e.g. '2015-2024'
    cohort_def TEXT NOT NULL,    -- serialized inclusion rules + code sets
    point REAL, ci_lo REAL, ci_hi REAL,
    n_pairs INTEGER, n_crashes INTEGER, n_persons INTEGER,
    PRIMARY KEY (run_ts, estimand)
);

-- One decode of the raw codes. Code sets are pinned literals from codebook.py:
--   belted   REST_USE in ({cb.sql_in(cb.REST_USE_BELTED)})  = shoulder-only, lap-only, lap+shoulder
--   frontal  IMPACT1 in ({cb.sql_in(cb.IMPACT1_FRONTAL_CORE)})  Forman 2019 core zone
--   frontal_wide IMPACT1 in ({cb.sql_in(cb.IMPACT1_FRONTAL_WIDE)})  sensitivity variant
--   light vehicle = passenger car + light truck/van (NHTSA grouping)
--   age {cb.AGE_MIN}-{cb.AGE_MAX} per Atwood/Noh/Craig 2023; FARS sentinels 998/999 excluded by the window
-- The view is dropped and recreated on every connect so definition changes
-- reach existing databases without a migration.
DROP VIEW IF EXISTS v_occupant;
CREATE VIEW v_occupant AS
SELECT p.source, p.year, p.st_case, p.veh_no, p.per_no,
       p.sex, p.age, p.seat_pos, v.mod_year, v.impact1, a.man_coll,
       CASE WHEN p.sex={cb.SEX_FEMALE} THEN 1 WHEN p.sex={cb.SEX_MALE} THEN 0 END AS is_female,
       CASE WHEN p.per_typ IN ({cb.sql_in(cb.PER_TYP_OCCUPANTS)}) THEN 1 ELSE 0 END AS is_occupant,
       CASE WHEN p.seat_pos IN ({cb.sql_in(cb.SEAT_POS_FRONT_OUTBOARD)}) THEN 1 ELSE 0 END AS is_front_outboard,
       CASE WHEN p.rest_use IN ({cb.sql_in(cb.REST_USE_BELTED)}) THEN 1 ELSE 0 END AS is_belted,
       CASE WHEN v.impact1 IN ({cb.sql_in(cb.IMPACT1_FRONTAL_CORE)}) THEN 1 ELSE 0 END AS is_frontal,
       CASE WHEN v.impact1 IN ({cb.sql_in(cb.IMPACT1_FRONTAL_WIDE)}) THEN 1 ELSE 0 END AS is_frontal_wide,
       CASE WHEN a.man_coll = {cb.MAN_COLL_HEAD_ON} THEN 1 ELSE 0 END AS is_head_on,
       CASE WHEN p.age BETWEEN {cb.AGE_MIN} AND {cb.AGE_MAX} THEN 1 ELSE 0 END AS age_in_range,
       CASE WHEN v.body_typ IN ({cb.sql_in(cb.BODY_TYP_LIGHT_VEHICLE)}) THEN 1 ELSE 0 END AS is_light_vehicle,
       CASE WHEN p.inj_sev={cb.INJ_SEV_FATAL} THEN 1
            WHEN p.inj_sev IN ({cb.sql_in(cb.INJ_SEV_NONFATAL)}) THEN 0 END AS died
FROM person p
JOIN vehicle  v USING (source, year, st_case, veh_no)
JOIN accident a USING (source, year, st_case)
WHERE p.per_typ IN ({cb.sql_in(cb.PER_TYP_OCCUPANTS)});
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def insert_frame(conn: sqlite3.Connection, table: str, columns: list[str],
                 rows: list[tuple]) -> int:
    """INSERT OR REPLACE rows into table. Returns the number of rows written."""
    placeholders = ", ".join("?" for _ in columns)
    before = conn.total_changes
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    return conn.total_changes - before
