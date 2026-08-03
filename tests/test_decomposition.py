"""The seat/sex split is the correctness-critical part: a pooled ratio that
silently carries the seat effect is the failure this whole module exists to
prevent. These tests plant known f and s values and check they come back."""

import math

from conftest import add_pairs

from crashgap.analysis import decompose, double_pair


def plant(conn, r_passenger: tuple[int, int], r_drives: tuple[int, int]) -> None:
    """Plant discordant counts: (she-died, he-died) in each seat configuration."""
    case = 1
    f_pax, m_pax = r_passenger
    f_drv, m_drv = r_drives
    case = add_pairs(conn, case, f_pax, female_seat=13, female_died=1, male_died=0)
    case = add_pairs(conn, case, m_pax, female_seat=13, female_died=0, male_died=1)
    case = add_pairs(conn, case, f_drv, female_seat=11, female_died=1, male_died=0)
    add_pairs(conn, case, m_drv, female_seat=11, female_died=0, male_died=1)


def test_recovers_planted_sex_and_seat_effects(conn):
    # she-passenger ratio = 24/12 = 2.0 = f*s ; she-drives = 18/36 = 0.5 = f/s
    # => f = sqrt(2.0 * 0.5) = 1.0 ; s = sqrt(2.0 / 0.5) = 2.0
    plant(conn, r_passenger=(24, 12), r_drives=(18, 36))
    dec = decompose(conn, (2022, 2022), reps=200)
    assert dec.rr_she_passenger == 2.0
    assert dec.rr_she_drives == 0.5
    assert math.isclose(dec.sex_effect, 1.0, abs_tol=1e-6)
    assert math.isclose(dec.seat_effect, 2.0, abs_tol=1e-6)


def test_pooled_ratio_is_confounded_by_seat(conn):
    """The regression this guards: pure seat effect, zero sex effect, and the
    pooled ratio still reads as a female penalty because seats are imbalanced."""
    # f = 1.0 exactly; s = 2.0; far more pairs where she is the passenger
    plant(conn, r_passenger=(80, 40), r_drives=(5, 10))
    dec = decompose(conn, (2022, 2022), reps=200)
    dp = double_pair(conn, (2022, 2022), reps=200)
    assert math.isclose(dec.sex_effect, 1.0, abs_tol=1e-6)     # no sex effect
    assert dp.rr > 1.5                                          # but pooled looks damning
    assert not dec.sex_is_significant


def test_symmetric_seats_leave_pooled_and_sex_agreeing(conn):
    # balanced seats, genuine female effect of 2.0, no seat effect
    plant(conn, r_passenger=(40, 20), r_drives=(40, 20))
    dec = decompose(conn, (2022, 2022), reps=300)
    dp = double_pair(conn, (2022, 2022), reps=300)
    assert math.isclose(dec.sex_effect, 2.0, abs_tol=1e-6)
    assert math.isclose(dec.seat_effect, 1.0, abs_tol=1e-6)
    assert math.isclose(dp.rr, 2.0, abs_tol=1e-6)
    assert dec.sex_is_significant


def test_significance_flags_track_the_interval(conn):
    plant(conn, r_passenger=(3, 3), r_drives=(3, 3))   # tiny, null, wide CI
    dec = decompose(conn, (2022, 2022), reps=300)
    assert math.isclose(dec.sex_effect, 1.0, abs_tol=1e-6)
    assert not dec.sex_is_significant
    assert not dec.seat_is_significant


def test_empty_strata_do_not_crash(conn):
    add_pairs(conn, 1, 4, female_seat=13, female_died=1, male_died=0)  # no she-drives
    dec = decompose(conn, (2022, 2022), reps=100)
    assert dec.rr_she_drives is None
    assert dec.sex_effect is None and dec.seat_effect is None
    assert dec.sex_ci == (None, None)


def test_results_table_gets_all_three_estimands(conn):
    from crashgap.analysis import analyze
    plant(conn, r_passenger=(24, 12), r_drives=(18, 36))
    analyze(conn, (2022, 2022), reps=100)
    estimands = {r[0] for r in conn.execute("SELECT estimand FROM results")}
    for variant in ("core", "wide", "headon"):
        assert f"fars_doublepair_sex_effect_frontal_{variant}" in estimands
        assert f"fars_doublepair_seat_effect_rightfront_frontal_{variant}" in estimands
        assert (f"fars_doublepair_pooled_f_vs_m_seatconfounded_frontal_{variant}"
                in estimands)
