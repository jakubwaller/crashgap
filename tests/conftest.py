from __future__ import annotations

import sqlite3

import pytest

from crashgap.db import connect


@pytest.fixture()
def conn(tmp_path) -> sqlite3.Connection:
    return connect(tmp_path / "test.db")


def add_pairs(conn: sqlite3.Connection, start_case: int, n: int, *,
              female_seat: int, female_died: int, male_died: int,
              year: int = 2022) -> int:
    """n identical mixed-sex pairs, one per crash. Returns the next free case id."""
    male_seat = 13 if female_seat == 11 else 11
    for i in range(n):
        add_crash(conn, start_case + i, [
            {"sex": 2, "seat_pos": female_seat, "inj_sev": 4 if female_died else 0,
             "per_typ": 1 if female_seat == 11 else 2},
            {"sex": 1, "seat_pos": male_seat, "inj_sev": 4 if male_died else 0,
             "per_typ": 1 if male_seat == 11 else 2},
        ], year=year)
    return start_case + n


def add_crash(conn: sqlite3.Connection, st_case: int, occupants: list[dict],
              year: int = 2022, man_coll: int = 0, body_typ: int = 4,
              mod_year: int = 2015, impact1: int = 12) -> None:
    """One crash, one vehicle, occupants as dicts of person-field overrides."""
    conn.execute(
        "INSERT OR REPLACE INTO accident (source, year, st_case, man_coll, fatals)"
        " VALUES ('fars', ?, ?, ?, 1)", (year, st_case, man_coll))
    conn.execute(
        "INSERT OR REPLACE INTO vehicle (source, year, st_case, veh_no, body_typ,"
        " mod_year, impact1) VALUES ('fars', ?, ?, 1, ?, ?, ?)",
        (year, st_case, body_typ, mod_year, impact1))
    for i, occ in enumerate(occupants, start=1):
        row = {"per_typ": 1 if i == 1 else 2, "sex": 1, "age": 40,
               "seat_pos": 11 if i == 1 else 13, "rest_use": 3, "inj_sev": 0}
        row.update(occ)
        conn.execute(
            "INSERT OR REPLACE INTO person (source, year, st_case, veh_no, per_no,"
            " per_typ, sex, age, seat_pos, rest_use, inj_sev)"
            " VALUES ('fars', ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)",
            (year, st_case, i, row["per_typ"], row["sex"], row["age"],
             row["seat_pos"], row["rest_use"], row["inj_sev"]))
    conn.commit()
