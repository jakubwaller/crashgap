"""Planted-effect tests for the v1 differenced conditional logit (design doc
section 7). y = 1 iff the FEMALE occupant died, and every covariate is her
value minus his - get that backwards anywhere and every odds ratio here
inverts. test_pooled_matches_v0_decompose is the tripwire: with no age
variation the pooled fit must equal v0's Evans geometric mean exactly, and a
reciprocal sign on x_seat or y would fail it loudly rather than silently."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from conftest import add_pairs, add_samesex_pairs

from crashgap.analysis import (
    MOD_YEAR_BANDS,
    POWER_FLOOR,
    analyze_v1,
    condlogit_bootstrap_ci,
    condlogit_by_band,
    condlogit_by_band_separate,
    condlogit_frame,
    condlogit_pooled,
    condlogit_trend,
    decompose,
    filter_headline_safe,
    occupant_frame,
    samesex_pairs_frame,
    samesex_seat_effect,
    samesex_seat_effect_ageadj,
    seatsex_interaction,
    seatsex_interaction_ageadj,
)
from crashgap.db import insert_frame

# ---------------------------------------------------------------------------
# Bulk synthetic-data helper for the tests that need real per-pair age
# variation (7.2, 7.4, 7.6, 7.9) - add_pairs only plants IDENTICAL pairs, so
# these bypass it and insert discordant rows directly, one INSERT OR REPLACE
# executemany per table instead of one add_crash + commit per pair.
# ---------------------------------------------------------------------------


def _bulk_pairs(conn, rows: list[dict], year: int = 2022) -> None:
    accident_rows, vehicle_rows, person_rows = [], [], []
    for r in rows:
        st_case = r["st_case"]
        accident_rows.append(("fars", year, st_case, 0, 1))
        vehicle_rows.append(("fars", year, st_case, 1, 4, r["mod_year"], 12))
        f_seat = r["f_seat"]
        m_seat = 13 if f_seat == 11 else 11
        f_per_typ = 1 if f_seat == 11 else 2
        m_per_typ = 1 if m_seat == 11 else 2
        person_rows.append(("fars", year, st_case, 1, 1, f_per_typ, 2,
                            r["f_age"], f_seat, 3, 4 if r["f_died"] else 0))
        person_rows.append(("fars", year, st_case, 1, 2, m_per_typ, 1,
                            r["m_age"], m_seat, 3, 4 if r["m_died"] else 0))
    insert_frame(conn, "accident",
                ["source", "year", "st_case", "man_coll", "fatals"], accident_rows)
    insert_frame(conn, "vehicle",
                ["source", "year", "st_case", "veh_no", "body_typ", "mod_year", "impact1"],
                vehicle_rows)
    insert_frame(conn, "person",
                ["source", "year", "st_case", "veh_no", "per_no", "per_typ", "sex", "age",
                 "seat_pos", "rest_use", "inj_sev"],
                person_rows)


def _logistic_discordant_rows(rng, n_candidates: int, start_case: int, mod_year: int,
                              intercept: float, slope: float,
                              f_age_fn, m_age_fn) -> tuple[list[dict], int]:
    """Draw n_candidates independent-occupant pairs, keep only the discordant
    ones (exactly one died), with death drawn from logit(p) = intercept +
    slope * age for each occupant independently. This is exactly the
    mechanism whose discordant f:m ratio equals the true odds ratio."""
    rows = []
    case = start_case
    for _ in range(n_candidates):
        f_age, m_age = f_age_fn(rng), m_age_fn(rng)
        f_seat = int(rng.choice([11, 13]))
        p_f = 1 / (1 + np.exp(-(intercept + slope * f_age)))
        p_m = 1 / (1 + np.exp(-(intercept + slope * m_age)))
        f_died = rng.random() < p_f
        m_died = rng.random() < p_m
        if f_died == m_died:
            continue
        rows.append({"st_case": case, "mod_year": mod_year, "f_seat": f_seat,
                     "f_age": f_age, "m_age": m_age,
                     "f_died": int(f_died), "m_died": int(m_died)})
        case += 1
    return rows, case


# ---------------------------------------------------------------------------
# 7.1 Exact-equivalence test (regression, against v0).
# ---------------------------------------------------------------------------


def test_pooled_matches_v0_decompose_when_age_is_flat(conn):
    """Same planted counts as test_decomposition's plant(r_passenger=(24,12),
    r_drives=(18,36)): with age_diff identically 0 for every pair, the
    differenced logit's pooled female coefficient must equal v0's
    decompose().sex_effect to tight tolerance - proves the reduction is
    correct, not a silently different quantity from v0's headline number."""
    case = 1
    case = add_pairs(conn, case, 24, female_seat=13, female_died=1, male_died=0)
    case = add_pairs(conn, case, 12, female_seat=13, female_died=0, male_died=1)
    case = add_pairs(conn, case, 18, female_seat=11, female_died=1, male_died=0)
    add_pairs(conn, case, 36, female_seat=11, female_died=0, male_died=1)

    dec = decompose(conn, (2022, 2022), reps=200)
    occ = occupant_frame(conn, (2022, 2022))
    fit = condlogit_pooled(occ)

    assert math.isclose(fit.odds_ratio("x_const"), dec.sex_effect, abs_tol=1e-6)
    # the seat coefficient reduces the same way and should agree too
    assert math.isclose(fit.odds_ratio("x_seat"), dec.seat_effect, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# 7.2 Planted age effect, no sex/seat effect.
# ---------------------------------------------------------------------------


def test_planted_age_effect_recovered_no_sex_or_seat_effect(conn):
    rng = np.random.default_rng(20260901)
    rows, _ = _logistic_discordant_rows(
        rng, 12000, start_case=1, mod_year=2015, intercept=-3, slope=0.04,
        f_age_fn=lambda r: r.uniform(16, 90), m_age_fn=lambda r: r.uniform(16, 90))
    assert len(rows) > 1000  # sanity: got a usable discordant sample
    _bulk_pairs(conn, rows)

    occ = occupant_frame(conn, (2022, 2022))
    fit = condlogit_pooled(occ)

    lo, hi = fit.coef_ci("x_age")
    assert lo is not None and lo <= 0.04 <= hi
    lo_c, hi_c = fit.coef_ci("x_age_curv")
    assert lo_c is not None and lo_c <= 0.0 <= hi_c
    assert math.isclose(fit.odds_ratio("x_const"), 1.0, abs_tol=0.15)


# ---------------------------------------------------------------------------
# 7.3 Planted band-varying sex effect (no age confound).
# ---------------------------------------------------------------------------

_BAND_PLAN = [
    ("2000-2009", 2005, 30, 10, 3.0),
    ("2010-2016", 2013, 30, 20, 1.5),
    ("2017-2026", 2020, 25, 25, 1.0),
]


def _plant_homogeneous_bands(conn) -> None:
    """Three bands, distinct planted female OR, age matched within every
    pair (no age-sex correlation) and BOTH seat configs planted with the
    same f:m ratio in every band (no seat effect anywhere) - the same
    same-ratio-both-seats trick test_decomposition uses for a null seat
    effect, applied per band."""
    case = 1
    for _label, mod_year, f, m, _target in _BAND_PLAN:
        case = add_pairs(conn, case, f, female_seat=13, female_died=1, male_died=0,
                         mod_year=mod_year)
        case = add_pairs(conn, case, m, female_seat=13, female_died=0, male_died=1,
                         mod_year=mod_year)
        case = add_pairs(conn, case, f, female_seat=11, female_died=1, male_died=0,
                         mod_year=mod_year)
        case = add_pairs(conn, case, m, female_seat=11, female_died=0, male_died=1,
                         mod_year=mod_year)


def test_band_varying_sex_effect_not_smeared_by_pooled_nuisance(conn):
    _plant_homogeneous_bands(conn)
    occ = occupant_frame(conn, (2022, 2022))
    pooled_bands = condlogit_by_band(occ)
    separate_bands = condlogit_by_band_separate(occ)
    for label, _, _, _, target in _BAND_PLAN:
        col = f"x_female_band_{label}"
        assert math.isclose(pooled_bands.odds_ratio(col), target, rel_tol=0.05)
        assert math.isclose(separate_bands[label].odds_ratio(col), target, rel_tol=0.05)


def test_pooled_and_bands_share_the_same_discordant_cohort(conn):
    """`_pooled` is meant to be the band-free collapse of the same model
    `_band_*` rows come from - both must be fit over the identical eligible/
    discordant cohort, or the two families silently stop reconciling (as
    happened when only "bands" mode filtered mod_year - see
    test_trend_slope_unaffected_by_mod_year_sentinel_rows)."""
    _plant_homogeneous_bands(conn)
    occ = occupant_frame(conn, (2022, 2022))
    pooled_fit = condlogit_pooled(occ)
    band_fit = condlogit_by_band(occ)
    assert pooled_fit.n_discordant_pairs == band_fit.n_discordant_pairs


# ---------------------------------------------------------------------------
# 7.4 The trend-question test - the publish-blocker's direct proof.
# ---------------------------------------------------------------------------


def test_age_adjustment_repairs_spurious_band_trend(conn):
    """Flat true female effect (OR=1.0 in every band), but women average
    8/4/0 years younger than their paired men across the three bands, and
    younger is protective (logit(p) = -3 + 0.05*age). The naive unadjusted
    ratio must show a spurious INCREASING trend (mirroring the real
    naive-data pattern); the age-adjusted band model must recover ~flat 1.0
    in every band."""
    rng = np.random.default_rng(20260902)
    band_gaps = [("2000-2009", 2005, 8), ("2010-2016", 2013, 4), ("2017-2026", 2020, 0)]
    case = 1
    all_rows = []
    for _label, mod_year, gap in band_gaps:
        rows, case = _logistic_discordant_rows(
            rng, 6000, start_case=case, mod_year=mod_year, intercept=-3, slope=0.05,
            f_age_fn=lambda r, g=gap: np.clip(r.uniform(20, 70) - g + r.normal(0, 4), 16, 96),
            m_age_fn=lambda r: r.uniform(20, 70))
        all_rows.extend(rows)
    _bulk_pairs(conn, all_rows)

    naive_ratios = []
    for _label, mod_year, _gap in band_gaps:
        band_rows = [r for r in all_rows if r["mod_year"] == mod_year]
        f_only = sum(1 for r in band_rows if r["f_died"])
        m_only = sum(1 for r in band_rows if r["m_died"])
        naive_ratios.append(f_only / m_only)
    assert naive_ratios[0] < naive_ratios[1] < naive_ratios[2]

    occ = occupant_frame(conn, (2022, 2022))
    band_fit = condlogit_by_band(occ)
    for label, _, _ in band_gaps:
        col = f"x_female_band_{label}"
        or_ = band_fit.odds_ratio(col)
        assert or_ is not None
        assert math.isclose(or_, 1.0, abs_tol=0.25)


# ---------------------------------------------------------------------------
# 7.5 Same-sex seat x sex interaction - null and true cases.
# ---------------------------------------------------------------------------


def test_samesex_interaction_null_case(conn):
    case = 1
    case = add_samesex_pairs(conn, case, 26, sex="male", driver_died=0, passenger_died=1)
    case = add_samesex_pairs(conn, case, 20, sex="male", driver_died=1, passenger_died=0)
    case = add_samesex_pairs(conn, case, 26, sex="female", driver_died=0, passenger_died=1)
    add_samesex_pairs(conn, case, 20, sex="female", driver_died=1, passenger_died=0)

    male = samesex_pairs_frame(conn, (2022, 2022), sex="male")
    female = samesex_pairs_frame(conn, (2022, 2022), sex="female")
    interaction = seatsex_interaction(male, female, reps=500)

    assert math.isclose(interaction.log_ratio, 0.0, abs_tol=1e-6)
    assert interaction.ci_lo <= 0.0 <= interaction.ci_hi


def test_samesex_interaction_true_effect_case(conn):
    case = 1
    case = add_samesex_pairs(conn, case, 24, sex="male", driver_died=0, passenger_died=1)
    case = add_samesex_pairs(conn, case, 20, sex="male", driver_died=1, passenger_died=0)
    case = add_samesex_pairs(conn, case, 40, sex="female", driver_died=0, passenger_died=1)
    add_samesex_pairs(conn, case, 20, sex="female", driver_died=1, passenger_died=0)

    male = samesex_pairs_frame(conn, (2022, 2022), sex="male")
    female = samesex_pairs_frame(conn, (2022, 2022), sex="female")
    interaction = seatsex_interaction(male, female, reps=1000)

    expected = math.log(2.0 / 1.2)
    assert math.isclose(interaction.log_ratio, expected, abs_tol=1e-4)
    assert interaction.ci_lo <= expected <= interaction.ci_hi


# ---------------------------------------------------------------------------
# 7.6 Continuous-trend model cross-check.
# ---------------------------------------------------------------------------


def test_continuous_trend_recovers_planted_slope(conn):
    case = 1
    years_list = list(range(1975, 2030, 5))
    mean_y = float(np.mean(years_list))
    slope_per_decade = -0.4
    n_per_config = 80
    for y in years_list:
        p = 1 / (1 + np.exp(-(slope_per_decade * (y - mean_y) / 10)))
        f_only = max(1, min(n_per_config - 1, round(n_per_config * p)))
        m_only = n_per_config - f_only
        for seat in (13, 11):
            case = add_pairs(conn, case, f_only, female_seat=seat, female_died=1,
                             male_died=0, mod_year=y)
            case = add_pairs(conn, case, m_only, female_seat=seat, female_died=0,
                             male_died=1, mod_year=y)

    occ = occupant_frame(conn, (2022, 2022))
    trend_fit = condlogit_trend(occ)
    lo, hi = trend_fit.coef_ci("x_trend")
    assert lo is not None and lo <= slope_per_decade <= hi

    # sign/rough-magnitude cross-check against the discrete per-band pattern
    band_fit = condlogit_by_band(occ)
    or_first = band_fit.odds_ratio("x_female_band_1970-1999")
    or_last = band_fit.odds_ratio("x_female_band_2017-2026")
    assert or_first is not None and or_last is not None and or_first > or_last


def test_trend_slope_unaffected_by_mod_year_sentinel_rows(conn):
    """"trend" mode uses raw mod_year as a numeric regressor, unlike "bands"
    mode, which only ever sees mod_year through _band_labels and so gets
    out-of-range values filtered for free. Plant the same clean trend data as
    above, fit it, then add FARS's 9999 "not reported" sentinel and a couple
    of pre-1970 implausible model years with a reversed sex pattern and
    refit - the recovered slope and n_discordant_pairs must be numerically
    unchanged, because occupant_frame excludes those rows at the SQL level
    before any band_mode sees them."""
    case = 1
    years_list = list(range(1975, 2030, 5))
    mean_y = float(np.mean(years_list))
    slope_per_decade = -0.4
    n_per_config = 80
    for y in years_list:
        p = 1 / (1 + np.exp(-(slope_per_decade * (y - mean_y) / 10)))
        f_only = max(1, min(n_per_config - 1, round(n_per_config * p)))
        m_only = n_per_config - f_only
        for seat in (13, 11):
            case = add_pairs(conn, case, f_only, female_seat=seat, female_died=1,
                             male_died=0, mod_year=y)
            case = add_pairs(conn, case, m_only, female_seat=seat, female_died=0,
                             male_died=1, mod_year=y)

    occ_clean = occupant_frame(conn, (2022, 2022))
    clean_fit = condlogit_trend(occ_clean)

    for mod_year in (9999, 9999, 9999, 1923, 1954):
        case = add_pairs(conn, case, 1, female_seat=13, female_died=0, male_died=1,
                         mod_year=mod_year)

    occ_contaminated = occupant_frame(conn, (2022, 2022))
    contaminated_fit = condlogit_trend(occ_contaminated)

    assert contaminated_fit.coef["x_trend"] == clean_fit.coef["x_trend"]
    assert contaminated_fit.n_discordant_pairs == clean_fit.n_discordant_pairs


# ---------------------------------------------------------------------------
# 7.7 Per-band-independent-nuisance sensitivity - homogeneous case.
# ---------------------------------------------------------------------------


def test_separate_nuisance_agrees_with_pooled_when_nuisance_is_homogeneous(conn):
    _plant_homogeneous_bands(conn)
    occ = occupant_frame(conn, (2022, 2022))
    pooled = condlogit_by_band(occ)
    separate = condlogit_by_band_separate(occ)
    for label, _, _, _, _target in _BAND_PLAN:
        col = f"x_female_band_{label}"
        p_or, s_or = pooled.odds_ratio(col), separate[label].odds_ratio(col)
        assert p_or is not None and s_or is not None
        assert math.isclose(p_or, s_or, rel_tol=0.1)


# ---------------------------------------------------------------------------
# 7.8 No cross-channel leakage.
# ---------------------------------------------------------------------------


def test_no_cross_channel_leakage_from_samesex_pairs(conn):
    case = 1
    case = add_pairs(conn, case, 24, female_seat=13, female_died=1, male_died=0)
    case = add_pairs(conn, case, 12, female_seat=13, female_died=0, male_died=1)
    case = add_pairs(conn, case, 18, female_seat=11, female_died=1, male_died=0)
    case = add_pairs(conn, case, 36, female_seat=11, female_died=0, male_died=1)

    occ_before = occupant_frame(conn, (2022, 2022))
    fit_before = condlogit_pooled(occ_before)

    case = add_samesex_pairs(conn, case, 15, sex="male", driver_died=1, passenger_died=0)
    add_samesex_pairs(conn, case, 15, sex="female", driver_died=0, passenger_died=1)

    occ_after = occupant_frame(conn, (2022, 2022))
    fit_after = condlogit_pooled(occ_after)

    assert fit_before.coef == fit_after.coef
    assert fit_before.n_discordant_pairs == fit_after.n_discordant_pairs


# ---------------------------------------------------------------------------
# 7.9 Method-consistency check.
# ---------------------------------------------------------------------------


def test_cluster_analytic_ci_agrees_with_bootstrap_ci(conn):
    rng = np.random.default_rng(20260903)
    rows, _ = _logistic_discordant_rows(
        rng, 6000, start_case=1, mod_year=2015, intercept=-3, slope=0.04,
        f_age_fn=lambda r: r.uniform(16, 90), m_age_fn=lambda r: r.uniform(16, 90))
    _bulk_pairs(conn, rows)

    occ = occupant_frame(conn, (2022, 2022))
    fit = condlogit_pooled(occ)
    cluster_lo, cluster_hi = fit.or_ci("x_age")
    assert cluster_lo is not None

    frame = condlogit_frame(occ, band_mode="pooled")
    boot_lo, boot_hi = condlogit_bootstrap_ci(frame, "x_age", reps=300, seed=7)
    assert boot_lo is not None

    # the two methods need not match bit for bit, but their intervals must
    # substantially overlap - this is the cross-check, not a duplicate fit.
    # A bare non-empty-intersection check would pass even for barely-touching,
    # clearly-inconsistent intervals, so also require the interval midpoints
    # to agree within a stated relative tolerance.
    assert boot_lo <= cluster_hi and cluster_lo <= boot_hi
    cluster_mid = (cluster_lo + cluster_hi) / 2
    boot_mid = (boot_lo + boot_hi) / 2
    assert math.isclose(cluster_mid, boot_mid, rel_tol=0.02)


# ---------------------------------------------------------------------------
# 7.10 Edge cases.
# ---------------------------------------------------------------------------


def test_empty_bands_and_cohorts_return_none(conn):
    occ = occupant_frame(conn, (2022, 2022))
    assert occ.empty
    fit = condlogit_by_band(occ)
    for label in ("1970-1999", "2000-2009", "2010-2016", "2017-2026"):
        assert fit.odds_ratio(f"x_female_band_{label}") is None

    male_frame = samesex_pairs_frame(conn, (2022, 2022), sex="male")
    female_frame = samesex_pairs_frame(conn, (2022, 2022), sex="female")
    assert samesex_seat_effect(male_frame, sex="male").rr is None
    interaction = seatsex_interaction(male_frame, female_frame)
    assert interaction.log_ratio is None
    assert interaction.ci_lo is None and interaction.ci_hi is None


def test_concordant_only_band_returns_none(conn):
    # both survive -> concordant, contributes nothing to the discordant fit
    add_pairs(conn, 1, 10, female_seat=13, female_died=0, male_died=0, mod_year=2005)
    occ = occupant_frame(conn, (2022, 2022))
    fit = condlogit_by_band(occ)
    assert fit.odds_ratio("x_female_band_2000-2009") is None


def test_concordant_only_samesex_cohort_returns_none(conn):
    add_samesex_pairs(conn, 1, 10, sex="male", driver_died=1, passenger_died=1)
    frame = samesex_pairs_frame(conn, (2022, 2022), sex="male")
    assert samesex_seat_effect(frame, sex="male").rr is None


# ---------------------------------------------------------------------------
# 7.11 Power floor enforcement.
# ---------------------------------------------------------------------------


def test_power_floor_filters_out_underpowered_rows():
    rows = [
        {"estimand": "fars_condlogit_sex_effect_frontal_core_band_2000-2009",
         "cohort_def": json.dumps({"n_discordant_pairs": POWER_FLOOR - 1}), "point": 1.5},
        {"estimand": "fars_condlogit_sex_effect_frontal_core_band_2010-2016",
         "cohort_def": json.dumps({"n_discordant_pairs": POWER_FLOOR}), "point": 1.4},
        {"estimand": "fars_condlogit_sex_effect_frontal_core_band_2017-2026",
         "cohort_def": json.dumps({"n_discordant_pairs": POWER_FLOOR + 500}), "point": 1.1},
        # v0 rows carry no discordant/concordant split - always headline-safe
        {"estimand": "fars_doublepair_sex_effect_frontal_core",
         "cohort_def": json.dumps({"design": "evans"}), "point": 1.1},
    ]
    results = pd.DataFrame(rows)

    kept = filter_headline_safe(results)

    assert set(kept["estimand"]) == {
        "fars_condlogit_sex_effect_frontal_core_band_2010-2016",
        "fars_condlogit_sex_effect_frontal_core_band_2017-2026",
        "fars_doublepair_sex_effect_frontal_core",
    }


# ---------------------------------------------------------------------------
# Age-confound in the same-sex channel: the two cohorts can differ in
# within-pair age structure (on real FARS, female-female pairs carry a much
# older passenger far more often), and that composition alone manufactures an
# apparent seat x sex interaction. Planted truth here: identical seat effect
# (OR 1.2) for both sexes, a pure age effect (0.05 log-odds per year of
# passenger-minus-driver age gap), and a female cohort loaded with
# old-passenger pairs. The unadjusted log-ratio must show the spurious
# interaction; the age-adjusted one must recover ~0.
# ---------------------------------------------------------------------------


def _plant_samesex_profile(conn, case, sex, drv_age, pax_age, n_pax_died, n_drv_died):
    case = add_samesex_pairs(conn, case, n_pax_died, sex=sex, driver_died=0,
                             passenger_died=1, driver_age=drv_age, passenger_age=pax_age)
    return add_samesex_pairs(conn, case, n_drv_died, sex=sex, driver_died=1,
                             passenger_died=0, driver_age=drv_age, passenger_age=pax_age)


def test_samesex_age_confound_removed_by_adjustment(conn):
    # counts = round(N * sigmoid(log(1.2) + 0.05 * (pax_age - drv_age)))
    case = 1
    case = _plant_samesex_profile(conn, case, "male", 40, 40, 65, 55)     # gap 0
    case = _plant_samesex_profile(conn, case, "male", 70, 50, 28, 62)     # gap -20
    case = _plant_samesex_profile(conn, case, "male", 30, 44, 64, 26)     # gap +14
    case = _plant_samesex_profile(conn, case, "female", 40, 40, 33, 27)   # gap 0
    case = _plant_samesex_profile(conn, case, "female", 40, 75, 140, 20)  # gap +35
    _plant_samesex_profile(conn, case, "female", 60, 40, 18, 42)          # gap -20

    male = samesex_pairs_frame(conn, (2022, 2022), sex="male")
    female = samesex_pairs_frame(conn, (2022, 2022), sex="female")

    unadjusted = seatsex_interaction(male, female, reps=200)
    assert unadjusted.log_ratio > 0.4  # the age composition alone fakes an interaction

    adjusted = seatsex_interaction_ageadj(male, female)
    assert adjusted is not None
    assert abs(adjusted.log_ratio) < 0.2
    assert adjusted.ci_lo <= 0.0 <= adjusted.ci_hi
    # and both cohorts' age-adjusted seat effects sit near the planted OR 1.2
    for frame in (male, female):
        fit = samesex_seat_effect_ageadj(frame)
        assert 1.0 < fit.odds_ratio("x_const") < 1.45


# ---------------------------------------------------------------------------
# Band boundaries are inclusive on both edges: a pair whose mod_year sits
# exactly on a band's lower bound must land in that band, not vanish. (A
# `>` in place of `>=` in the band labeller passes every other test in this
# file - planted mod_years never sit on a lower edge - while silently
# dropping thousands of real pairs.)
# ---------------------------------------------------------------------------


def test_band_lower_boundary_pair_is_included(conn):
    case = 1
    for mod_year in (2000, 2010, 2017):
        case = add_pairs(conn, case, 1, female_seat=13, female_died=1, male_died=0,
                         mod_year=mod_year)

    occ = occupant_frame(conn, (2022, 2022))
    frame = condlogit_frame(occ, band_mode="bands")

    for label in ("2000-2009", "2010-2016", "2017-2026"):
        col = f"x_female_band_{label}"
        assert col in frame.columns
        assert frame[col].sum() == 1.0
    assert len(frame) == 3  # nothing dropped by the band labeller


# ---------------------------------------------------------------------------
# analyze_v1 end-to-end: the estimand-name and cohort_def contract the
# dashboard reads (a silently renamed estimand or dropped cohort_def key
# would break the power floor without failing any unit test above).
# ---------------------------------------------------------------------------


def test_analyze_v1_writes_the_full_estimand_contract(conn):
    _plant_homogeneous_bands(conn)
    case = 9000
    case = add_samesex_pairs(conn, case, 40, sex="male", driver_died=0, passenger_died=1)
    case = add_samesex_pairs(conn, case, 34, sex="male", driver_died=1, passenger_died=0)
    case = add_samesex_pairs(conn, case, 40, sex="female", driver_died=0, passenger_died=1)
    add_samesex_pairs(conn, case, 34, sex="female", driver_died=1, passenger_died=0)

    run_ts = analyze_v1(conn, (2022, 2022), reps=50)

    rows = pd.read_sql_query(
        "SELECT estimand, cohort_def FROM results WHERE run_ts = ?", conn, params=(run_ts,))
    bands = [f"{lo}-{hi}" for lo, hi in MOD_YEAR_BANDS]
    per_variant = (
        [f"sex_effect_frontal_{{v}}_band_{b}" for b in bands]
        + [f"sex_effect_frontal_{{v}}_band_{b}_separatenuisance" for b in bands]
        + [f"sex_effect_frontal_{{v}}_band_{b}_agepiecewise" for b in bands]
        + ["sex_effect_frontal_{v}_pooled", "seat_effect_rightfront_frontal_{v}",
           "age_slope_frontal_{v}", "age_curvature_frontal_{v}",
           "sex_trend_slope_frontal_{v}"]
    )
    expected = set()
    for v in ("core", "wide", "headon"):
        expected.update("fars_condlogit_" + name.format(v=v) for name in per_variant)
        expected.update(f"fars_samesex_{name}_frontal_{v}" for name in (
            "seat_effect_male", "seat_effect_female", "seatsex_interaction",
            "seat_effect_male_ageadj", "seat_effect_female_ageadj",
            "seatsex_interaction_ageadj"))
    assert set(rows["estimand"]) == expected
    assert len(rows) == len(expected)

    for cohort_def in rows["cohort_def"]:
        info = json.loads(cohort_def)
        assert "n_pairs" in info and "n_discordant_pairs" in info

    # additive only: no v0 rows appear under a v1 run_ts
    assert not rows["estimand"].str.startswith("fars_doublepair").any()
