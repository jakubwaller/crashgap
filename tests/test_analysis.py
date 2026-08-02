from conftest import add_crash

from crashgap.analysis import by_model_year_band, double_pair, pairs_frame


def test_discordant_ratio(conn):
    # 3 crashes where she died and he survived, 2 the reverse, 1 concordant
    for case in (1, 2, 3):
        add_crash(conn, case, [{"sex": 2, "inj_sev": 4}, {"sex": 1, "inj_sev": 0}])
    for case in (4, 5):
        add_crash(conn, case, [{"sex": 1, "inj_sev": 4}, {"sex": 2, "inj_sev": 0}])
    add_crash(conn, 6, [{"sex": 1, "inj_sev": 4}, {"sex": 2, "inj_sev": 4}])
    dp = double_pair(conn, (2022, 2022), reps=200)
    assert (dp.f_only, dp.m_only, dp.n_pairs) == (3, 2, 6)
    assert dp.rr == 1.5
    assert dp.ci_lo is not None and dp.ci_lo <= 1.5 <= dp.ci_hi


def test_pair_restriction_excludes_non_mixed_vehicles(conn):
    add_crash(conn, 1, [{"sex": 1}, {"sex": 1}])            # same-sex pair
    add_crash(conn, 2, [{"sex": 2}])                        # single occupant
    add_crash(conn, 3, [{"sex": 1}, {"sex": 2},             # three eligible
                        {"sex": 2, "seat_pos": 13, "per_typ": 2}])
    assert pairs_frame(conn, (2022, 2022)).empty


def test_ineligible_occupants_drop_out(conn):
    # unbelted male -> vehicle no longer holds an eligible pair
    add_crash(conn, 1, [{"sex": 2, "inj_sev": 4}, {"sex": 1, "rest_use": 20}])
    # rear seat, non-frontal impact, heavy vehicle, age outside window
    add_crash(conn, 2, [{"sex": 2, "seat_pos": 21}, {"sex": 1}])
    add_crash(conn, 3, [{"sex": 2}, {"sex": 1}], impact1=6)
    add_crash(conn, 4, [{"sex": 2}, {"sex": 1}], body_typ=50)
    add_crash(conn, 5, [{"sex": 2, "age": 12}, {"sex": 1}])
    # unknown outcome (inj_sev=9) is not counted as survived
    add_crash(conn, 6, [{"sex": 2, "inj_sev": 4}, {"sex": 1, "inj_sev": 9}])
    assert pairs_frame(conn, (2022, 2022)).empty


def test_st_case_not_confused_across_years(conn):
    # same st_case number in two years must be two distinct vehicles
    add_crash(conn, 7, [{"sex": 2, "inj_sev": 4}, {"sex": 1}], year=2021)
    add_crash(conn, 7, [{"sex": 1, "inj_sev": 4}, {"sex": 2}], year=2022)
    dp = double_pair(conn, (2021, 2022), reps=200)
    assert (dp.f_only, dp.m_only, dp.n_pairs, dp.n_crashes) == (1, 1, 2, 2)


def test_frontal_variants(conn):
    add_crash(conn, 1, [{"sex": 2, "inj_sev": 4}, {"sex": 1}], impact1=2)
    assert pairs_frame(conn, (2022, 2022), "core").empty
    assert len(pairs_frame(conn, (2022, 2022), "wide")) == 1
    add_crash(conn, 2, [{"sex": 2, "inj_sev": 4}, {"sex": 1}],
              impact1=12, man_coll=2)
    assert len(pairs_frame(conn, (2022, 2022), "headon")) == 1


def test_model_year_bands(conn):
    add_crash(conn, 1, [{"sex": 2, "inj_sev": 4}, {"sex": 1}], mod_year=2005)
    add_crash(conn, 2, [{"sex": 1, "inj_sev": 4}, {"sex": 2}], mod_year=2020)
    bands = by_model_year_band(conn, (2022, 2022), reps=50)
    by_label = bands.set_index("mod_year_band")
    assert by_label.loc["2000-2009", "f_only"] == 1
    assert by_label.loc["2017-2026", "m_only"] == 1
