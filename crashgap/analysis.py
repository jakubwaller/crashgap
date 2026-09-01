"""The Evans double-pair estimate, seat/sex decomposition, and clustered CIs.

FARS is a census of fatal crashes only, so a raw pooled odds ratio is
dominated by crash-involvement selection and can point the wrong way. The
double-pair design restricts to vehicles carrying exactly one eligible male
and one eligible female, so crash severity, delta-V, vehicle and impact
direction are held physically constant within the vehicle. Among discordant
pairs (exactly one of the two died), the female:male relative fatality risk
is (female died, male survived) / (male died, female survived).

**Seat position is the confound the pooled ratio hides.** It is the one
attribute that structurally differs within every pair, and it is ~73/27
collinear with sex (she is the right-front passenger far more often than she
drives). Measured on FARS 2015-2024, the pooled ratio splits into a large
right-front-seat penalty and a small, non-significant female effect - so the
pooled number is mostly a seat number wearing a sex label. Never report it
alone.

Evans' fix, and what this module reports: run the double pair separately in
the two seat configurations and take geometric means. With f the female
multiplier and s the right-front multiplier,

    R_she_passenger = f * s        R_she_drives = f / s
    sex effect  = sqrt(R_she_passenger * R_she_drives)   (seat cancels)
    seat effect = sqrt(R_she_passenger / R_she_drives)   (sex cancels)

A crash can contribute more than one vehicle pair, so occupants are not
independent: every CI is a percentile bootstrap resampling whole crashes
(year, st_case), never persons.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from . import codebook as cb

FRONTAL_PREDICATES = {
    "core": "is_frontal = 1",
    "wide": "is_frontal_wide = 1",
    "headon": "is_head_on = 1",
}

PAIRS_SQL = f"""
WITH occ AS (
    SELECT year, st_case, veh_no, is_female, died, mod_year, seat_pos
    FROM v_occupant
    WHERE source = :source AND year BETWEEN :y0 AND :y1
      AND is_occupant = 1 AND is_front_outboard = 1
      AND is_belted = 1 AND {{frontal}} AND is_light_vehicle = 1
      AND age_in_range = 1 AND is_female IS NOT NULL AND died IS NOT NULL
),
pairs AS (
    -- vehicles holding EXACTLY one eligible female + one eligible male
    SELECT year, st_case, veh_no,
           MAX(CASE WHEN is_female = 1 THEN died END) AS f_died,
           MAX(CASE WHEN is_female = 0 THEN died END) AS m_died,
           MAX(CASE WHEN is_female = 1 THEN seat_pos END) AS f_seat,
           MAX(mod_year) AS mod_year
    FROM occ
    GROUP BY year, st_case, veh_no
    HAVING SUM(is_female) = 1 AND SUM(1 - is_female) = 1
)
-- one of the two occupies seat {cb.SEAT_POS_DRIVER}, the other {cb.SEAT_POS_RIGHT_FRONT};
-- f_seat therefore identifies the seat configuration of the whole pair
SELECT year, st_case, veh_no, f_died, m_died, f_seat, mod_year FROM pairs
"""


@dataclass
class DoublePair:
    frontal: str
    years: tuple[int, int]
    f_only: int          # female died, male survived
    m_only: int          # male died, female survived
    n_pairs: int         # all mixed-sex pairs (incl. concordant)
    n_crashes: int       # distinct crashes contributing pairs
    rr: float | None     # f_only / m_only
    ci_lo: float | None
    ci_hi: float | None

    @property
    def n_persons(self) -> int:
        return 2 * self.n_pairs


def pairs_frame(conn: sqlite3.Connection, years: tuple[int, int],
                frontal: str = "core", source: str = "fars") -> pd.DataFrame:
    sql = PAIRS_SQL.format(frontal=FRONTAL_PREDICATES[frontal])
    return pd.read_sql_query(
        sql, conn, params={"source": source, "y0": years[0], "y1": years[1]})


def discordant_counts(pairs: pd.DataFrame) -> tuple[int, int]:
    """(she died & he survived, he died & she survived)."""
    f_only = int(((pairs.f_died == 1) & (pairs.m_died == 0)).sum())
    m_only = int(((pairs.m_died == 1) & (pairs.f_died == 0)).sum())
    return f_only, m_only


def _per_crash(pairs: pd.DataFrame, columns: dict[str, pd.Series]) -> np.ndarray:
    """Sum the given indicator columns per crash. Returns an (n_crashes, k) array."""
    frame = pairs.assign(**columns)
    grouped = frame.groupby(["year", "st_case"])[list(columns)].sum()
    return grouped.to_numpy(dtype=float)


def _indicators(pairs: pd.DataFrame) -> dict[str, pd.Series]:
    f_only = (pairs.f_died == 1) & (pairs.m_died == 0)
    m_only = (pairs.m_died == 1) & (pairs.f_died == 0)
    she_drives = pairs.f_seat == cb.SEAT_POS_DRIVER
    return {
        "f_only": f_only.astype(int),
        "m_only": m_only.astype(int),
        "f_only_drv": (f_only & she_drives).astype(int),
        "m_only_drv": (m_only & she_drives).astype(int),
        "f_only_pax": (f_only & ~she_drives).astype(int),
        "m_only_pax": (m_only & ~she_drives).astype(int),
    }


def _bootstrap_crashes(matrix: np.ndarray, reps: int, seed: int):
    """Yield resampled column sums, resampling whole crashes with replacement."""
    n = len(matrix)
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        yield counts @ matrix


def cluster_bootstrap_ci(pairs: pd.DataFrame, reps: int = 2000,
                         seed: int = 20260802) -> tuple[float | None, float | None]:
    """Percentile CI for f_only/m_only, resampling whole crashes."""
    ind = _indicators(pairs)
    matrix = _per_crash(pairs, {k: ind[k] for k in ("f_only", "m_only")})
    if len(matrix) == 0:
        return None, None
    ratios = [f / m for f, m in _bootstrap_crashes(matrix, reps, seed) if m > 0]
    if not ratios:
        return None, None
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def double_pair(conn: sqlite3.Connection, years: tuple[int, int],
                frontal: str = "core", reps: int = 2000) -> DoublePair:
    """The POOLED ratio - seat-confounded, never report it on its own.

    Kept because it is what the literature's raw double-pair line looks like
    and because the decomposition below is only interpretable next to it.
    """
    pairs = pairs_frame(conn, years, frontal)
    f_only, m_only = discordant_counts(pairs)
    rr = round(f_only / m_only, 3) if m_only > 0 else None
    ci_lo, ci_hi = cluster_bootstrap_ci(pairs, reps=reps)
    return DoublePair(
        frontal=frontal, years=years, f_only=f_only, m_only=m_only,
        n_pairs=len(pairs),
        n_crashes=pairs.groupby(["year", "st_case"]).ngroups,
        rr=rr, ci_lo=ci_lo, ci_hi=ci_hi,
    )


@dataclass
class Decomposition:
    """Evans geometric-mean split of the pooled ratio into sex and seat."""
    frontal: str
    years: tuple[int, int]
    rr_she_passenger: float | None   # f * s
    rr_she_drives: float | None      # f / s
    n_pairs_she_passenger: int
    n_pairs_she_drives: int
    sex_effect: float | None         # sqrt(R_pax * R_drv)
    sex_ci: tuple[float | None, float | None]
    seat_effect: float | None        # sqrt(R_pax / R_drv)
    seat_ci: tuple[float | None, float | None]
    n_pairs: int
    n_crashes: int

    @property
    def sex_is_significant(self) -> bool:
        lo, hi = self.sex_ci
        return lo is not None and (lo > 1.0 or hi < 1.0)

    @property
    def seat_is_significant(self) -> bool:
        lo, hi = self.seat_ci
        return lo is not None and (lo > 1.0 or hi < 1.0)


def decompose(conn: sqlite3.Connection, years: tuple[int, int],
              frontal: str = "core", reps: int = 2000,
              seed: int = 20260803) -> Decomposition:
    """Split the pooled ratio into a sex effect and a right-front-seat effect.

    Both strata are resampled inside ONE bootstrap replicate, so the two
    estimates share the resampled crashes and their intervals stay mutually
    consistent.
    """
    pairs = pairs_frame(conn, years, frontal)
    ind = _indicators(pairs)
    drv = pairs[pairs.f_seat == cb.SEAT_POS_DRIVER]
    pax = pairs[pairs.f_seat != cb.SEAT_POS_DRIVER]

    def ratio(subset: pd.DataFrame) -> float | None:
        f_only, m_only = discordant_counts(subset)
        return f_only / m_only if m_only > 0 else None

    r_drv, r_pax = ratio(drv), ratio(pax)
    sex = seat = None
    if r_drv and r_pax:
        sex = round(float(np.sqrt(r_pax * r_drv)), 4)
        seat = round(float(np.sqrt(r_pax / r_drv)), 4)

    cols = ("f_only_pax", "m_only_pax", "f_only_drv", "m_only_drv")
    matrix = _per_crash(pairs, {k: ind[k] for k in cols})
    sex_draws, seat_draws = [], []
    if len(matrix):
        for fp, mp, fd, md in _bootstrap_crashes(matrix, reps, seed):
            if mp > 0 and md > 0 and fp > 0 and fd > 0:
                rp, rd = fp / mp, fd / md
                sex_draws.append(np.sqrt(rp * rd))
                seat_draws.append(np.sqrt(rp / rd))

    def ci(draws: list[float]) -> tuple[float | None, float | None]:
        if not draws:
            return None, None
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return round(float(lo), 4), round(float(hi), 4)

    return Decomposition(
        frontal=frontal, years=years,
        rr_she_passenger=round(r_pax, 3) if r_pax else None,
        rr_she_drives=round(r_drv, 3) if r_drv else None,
        n_pairs_she_passenger=len(pax), n_pairs_she_drives=len(drv),
        sex_effect=sex, sex_ci=ci(sex_draws),
        seat_effect=seat, seat_ci=ci(seat_draws),
        n_pairs=len(pairs),
        n_crashes=pairs.groupby(["year", "st_case"]).ngroups,
    )


MOD_YEAR_BANDS = [(1970, 1999), (2000, 2009), (2010, 2016), (2017, 2026)]

# The dashboard's headline window: the newest MODERN_WINDOW calendar years of
# whatever is ingested. The full ingested span exists to power the model-year
# trend question; the headline "does being female raise fatality risk in the
# same crash" claim stays pinned to the modern decade so it never quietly
# absorbs 1990s vehicles as more history is ingested.
MODERN_WINDOW = 10


def modern_span(years: tuple[int, int]) -> tuple[int, int]:
    """The last MODERN_WINDOW calendar years of an ingested (y0, y1) span."""
    y0, y1 = years
    return (max(y0, y1 - MODERN_WINDOW + 1), y1)


def by_model_year_band(conn: sqlite3.Connection, years: tuple[int, int],
                       frontal: str = "core", reps: int = 1000) -> pd.DataFrame:
    """RR per vehicle model-year band, with crash-clustered CIs.

    Checked 2026-08-02 on FARS 2015-2024: all band CIs overlap, so this
    window is too narrow (and unadjusted for occupant age) to resolve the
    Atwood/Noh/Craig shrinking trend in either direction. Age adjustment
    is the v1 conditional-logit job - do not read a trend off these bands.
    """
    pairs = pairs_frame(conn, years, frontal)
    rows = []
    for lo, hi in MOD_YEAR_BANDS:
        band = pairs[(pairs.mod_year >= lo) & (pairs.mod_year <= hi)]
        f_only = int(((band.f_died == 1) & (band.m_died == 0)).sum())
        m_only = int(((band.m_died == 1) & (band.f_died == 0)).sum())
        ci_lo, ci_hi = cluster_bootstrap_ci(band, reps=reps)
        rows.append({
            "mod_year_band": f"{lo}-{hi}",
            "f_only": f_only, "m_only": m_only, "n_pairs": len(band),
            "rr": round(f_only / m_only, 3) if m_only > 0 else None,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
        })
    return pd.DataFrame(rows)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def write_result(conn: sqlite3.Connection, run_ts: str, estimand: str,
                 years: tuple[int, int], frontal: str,
                 point: float | None, ci: tuple[float | None, float | None],
                 n_pairs: int, n_crashes: int) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO results
           (run_ts, git_commit, estimand, fars_years, cohort_def,
            point, ci_lo, ci_hi, n_pairs, n_crashes, n_persons)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_ts, git_commit(), estimand,
         f"{years[0]}-{years[1]}", cb.cohort_def(frontal),
         point, ci[0], ci[1], n_pairs, n_crashes, 2 * n_pairs),
    )
    conn.commit()


def analyze(conn: sqlite3.Connection, years: tuple[int, int],
            reps: int = 2000) -> list[tuple[DoublePair, Decomposition]]:
    """Every frontal variant, pooled + decomposed, under one run_ts.

    Three estimands land per variant: the seat-confounded pooled ratio (kept
    for continuity and explicitly named `_seatconfounded`), the seat-balanced
    sex effect, and the right-front-seat effect.
    """
    run_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = []
    for frontal in ("core", "wide", "headon"):
        dp = double_pair(conn, years, frontal, reps=reps)
        dec = decompose(conn, years, frontal, reps=reps)
        write_result(conn, run_ts,
                     f"fars_doublepair_pooled_f_vs_m_seatconfounded_frontal_{frontal}",
                     years, frontal, dp.rr, (dp.ci_lo, dp.ci_hi),
                     dp.n_pairs, dp.n_crashes)
        write_result(conn, run_ts, f"fars_doublepair_sex_effect_frontal_{frontal}",
                     years, frontal, dec.sex_effect, dec.sex_ci,
                     dec.n_pairs, dec.n_crashes)
        write_result(conn, run_ts, f"fars_doublepair_seat_effect_rightfront_frontal_{frontal}",
                     years, frontal, dec.seat_effect, dec.seat_ci,
                     dec.n_pairs, dec.n_crashes)
        out.append((dp, dec))
    return out


# --------------------------------------------------------------------------
# v1: within-vehicle differenced conditional logit.
#
# For a 2-occupant stratum the true conditional MLE has a closed form: the
# stratum likelihood depends only on the DIFFERENCE of the linear predictor
# between the two occupants, so fitting reduces to an intercept-free logistic
# regression of y = 1{female occupant died} on covariates defined as HER
# VALUE MINUS HIS, over discordant pairs only (concordant pairs carry zero
# information, exactly as in v0's double_pair). Get an orientation backwards
# anywhere in this section - y, x_seat, or an age difference - and every
# odds ratio downstream inverts; test_pooled_matches_v0_decompose is the
# tripwire for that (it fails loudly if x_seat's sign is flipped).
#
# No standalone female covariate: within a discordant mixed-sex pair her
# indicator minus his is identically 1, so it would be collinear with the
# intercept. The female effect is instead carried by the band-dummy /
# constant / trend columns `condlogit_frame` builds per `band_mode` - see the
# design doc section 2 for why a female x seat interaction is unidentified
# from these pairs at all (section 6) and has to be routed through the
# separate same-sex channel below instead.
# --------------------------------------------------------------------------

OCCUPANT_SQL = """
WITH occ AS (
    SELECT year, st_case, veh_no, is_female, died, age, mod_year, seat_pos
    FROM v_occupant
    WHERE source = :source AND year BETWEEN :y0 AND :y1
      AND is_occupant = 1 AND is_front_outboard = 1
      AND is_belted = 1 AND {frontal} AND is_light_vehicle = 1
      AND age_in_range = 1 AND is_female IS NOT NULL AND died IS NOT NULL
      AND mod_year BETWEEN :myr_lo AND :myr_hi
),
eligible AS (
    -- vehicles holding EXACTLY one eligible female + one eligible male
    SELECT year, st_case, veh_no
    FROM occ
    GROUP BY year, st_case, veh_no
    HAVING SUM(is_female) = 1 AND SUM(1 - is_female) = 1
)
SELECT occ.year, occ.st_case, occ.veh_no, occ.is_female, occ.died, occ.age,
       occ.mod_year, occ.seat_pos
FROM occ JOIN eligible USING (year, st_case, veh_no)
"""


def occupant_frame(conn: sqlite3.Connection, years: tuple[int, int],
                   frontal: str = "core", source: str = "fars") -> pd.DataFrame:
    """Occupant-grain rows (two per vehicle) for the same eligible vehicles
    `pairs_frame` selects, at person grain instead of pre-aggregated, so age
    and seat position travel with each occupant for the differenced fit.

    mod_year is windowed to MOD_YEAR_BANDS' span here, at the SQL level, so
    every band_mode sees the same cohort - not just "bands" (which happens to
    exclude out-of-band years via _band_labels). Without this, FARS's 9999
    "not reported" sentinel and a handful of implausible pre-1970 model years
    act as extreme-leverage points in "trend"/"pooled" mode, which use
    mod_year as a numeric regressor rather than a label lookup."""
    sql = OCCUPANT_SQL.format(frontal=FRONTAL_PREDICATES[frontal])
    return pd.read_sql_query(
        sql, conn, params={
            "source": source, "y0": years[0], "y1": years[1],
            "myr_lo": MOD_YEAR_BANDS[0][0], "myr_hi": MOD_YEAR_BANDS[-1][1],
        })


def _pivot_pairs(occ: pd.DataFrame) -> pd.DataFrame:
    """occupant_frame -> one row per vehicle, her/his columns side by side."""
    keys = ["year", "st_case", "veh_no"]
    f = occ[occ.is_female == 1].set_index(keys)
    m = occ[occ.is_female == 0].set_index(keys)
    return f.join(m, how="inner", lsuffix="_f", rsuffix="_m").reset_index()


def age_diff_quadratic(age_f: np.ndarray, age_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x_age = her age minus his; x_age_curv = x_age * mean(age_f, age_m),
    which reproduces the age^2 curvature term (age_f^2 - age_m^2 =
    x_age * (age_f + age_m)) at half scale - a constant rescaling of the
    fitted coefficient, not a different quantity."""
    age_f = np.asarray(age_f, dtype=float)
    age_m = np.asarray(age_m, dtype=float)
    x_age = age_f - age_m
    return x_age, x_age * (age_f + age_m) / 2


def age_diff_piecewise(age_f: np.ndarray, age_m: np.ndarray,
                       knot: int = 65) -> tuple[np.ndarray, np.ndarray]:
    """Two-piece linear spline in age (hinge at `knot`), differenced the same
    way as the quadratic form - a functional-form sensitivity check for the
    threshold effects a quadratic can't capture."""
    age_f = np.asarray(age_f, dtype=float)
    age_m = np.asarray(age_m, dtype=float)
    below_f, above_f = np.minimum(age_f, knot), np.maximum(age_f - knot, 0)
    below_m, above_m = np.minimum(age_m, knot), np.maximum(age_m - knot, 0)
    return below_f - below_m, above_f - above_m


_AGE_BUILDERS = {
    "quadratic": ("x_age", "x_age_curv", age_diff_quadratic),
    "piecewise": ("x_age_below", "x_age_above", age_diff_piecewise),
}


def _band_labels(mod_year: np.ndarray) -> np.ndarray:
    """mod_year -> the matching MOD_YEAR_BANDS label, or None if it falls in
    no band (object array so None survives alongside string labels)."""
    labels = np.full(len(mod_year), None, dtype=object)
    for lo, hi in MOD_YEAR_BANDS:
        labels[(mod_year >= lo) & (mod_year <= hi)] = f"{lo}-{hi}"
    return labels


def condlogit_frame(occ: pd.DataFrame, band_mode: str,
                    age_form: str = "quadratic") -> pd.DataFrame:
    """occupant_frame -> a fit-ready frame: y plus differenced covariates,
    restricted to discordant mixed-sex pairs (concordant pairs carry zero
    information for a matched-pairs likelihood and are dropped here, same as
    v0). y = 1 iff the FEMALE occupant died; every covariate is her value
    minus his - orientation is load-bearing, see the module-level note above.

    band_mode selects how the female effect enters (never fit jointly - band
    is a coarsened function of year and the two are collinear together):
      "bands"  one dummy per model-year band actually present in the data
               (the default trend model - beta_seat/age pooled across bands)
      "trend"  x_const=1 plus x_trend, a continuous per-decade slope
      "pooled" a single x_const=1 column (validation target vs v0, see
               test_pooled_matches_v0_decompose)
    """
    if age_form not in _AGE_BUILDERS:
        raise ValueError(f"unknown age_form: {age_form!r}")
    col_lo, col_hi, builder = _AGE_BUILDERS[age_form]

    paired = _pivot_pairs(occ)
    paired = paired[(paired.died_f + paired.died_m) == 1]
    if paired.empty:
        return pd.DataFrame(columns=["year", "st_case", "y", "x_seat", col_lo, col_hi])

    x_seat = np.where(paired["seat_pos_f"].to_numpy() == cb.SEAT_POS_RIGHT_FRONT, 1.0, -1.0)
    lo, hi = builder(paired["age_f"].to_numpy(), paired["age_m"].to_numpy())
    frame = pd.DataFrame({
        "year": paired["year"].to_numpy(), "st_case": paired["st_case"].to_numpy(),
        "y": paired["died_f"].astype(int).to_numpy(), "x_seat": x_seat,
        col_lo: lo, col_hi: hi,
    })

    mod_year = paired["mod_year_f"].to_numpy()
    if band_mode == "pooled":
        frame["x_const"] = 1.0
    elif band_mode == "trend":
        frame["x_const"] = 1.0
        frame["x_trend"] = (mod_year - mod_year.mean()) / 10
    elif band_mode == "bands":
        labels = _band_labels(mod_year)
        mask = pd.notna(labels)
        frame = frame[mask]
        dummies = pd.get_dummies(labels[mask], prefix="x_female_band", dtype=float)
        dummies.index = frame.index
        frame = pd.concat([frame, dummies], axis=1)
    else:
        raise ValueError(f"unknown band_mode: {band_mode!r}")
    return frame.reset_index(drop=True)


_META_COLS = {"year", "st_case", "y"}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in _META_COLS]


@dataclass
class CondLogit:
    """One differenced-logit fit, mirroring DoublePair/Decomposition's shape.
    `coef`/`se` are raw log-odds; `odds_ratio`/`or_ci` exponentiate. A
    feature that was identically zero in this frame (no discordant pair
    varies on it - e.g. age when every planted pair is age-matched) still
    gets a coef of 0.0 with no CI, so callers can look any expected column up
    without a KeyError; it is otherwise excluded from the fit rather than
    left in, since a zero column carries no likelihood information and
    singularizes the fit rather than merely costing precision."""
    feature_cols: tuple[str, ...]
    coef: dict[str, float]
    se: dict[str, float]
    n_pairs: int              # eligible mixed-sex pairs in the source frame
    n_discordant_pairs: int   # informative (discordant) rows the fit used
    n_crashes: int             # crashes among the eligible pairs
    converged: bool

    def coef_ci(self, name: str, z: float = 1.959963985) -> tuple[float | None, float | None]:
        b, s = self.coef.get(name), self.se.get(name)
        if b is None or s is None or not np.isfinite(s):
            return None, None
        return round(b - z * s, 6), round(b + z * s, 6)

    def odds_ratio(self, name: str) -> float | None:
        b = self.coef.get(name)
        return round(float(np.exp(b)), 4) if b is not None else None

    def or_ci(self, name: str) -> tuple[float | None, float | None]:
        lo, hi = self.coef_ci(name)
        if lo is None:
            return None, None
        return round(float(np.exp(lo)), 4), round(float(np.exp(hi)), 4)


def fit_condlogit(frame: pd.DataFrame, n_pairs: int) -> CondLogit:
    """Fit the intercept-free differenced logit; cluster-robust SEs on
    (year, st_case), the same clustering `cluster_bootstrap_ci` resamples.
    Empty input, too few discordant rows for the columns present, or a fit
    that fails to converge (including perfect separation) all return a
    `CondLogit` with an empty `coef` rather than raising - callers read that
    back as None estimands (see test_empty_bands_and_cohorts_return_none)."""
    all_cols = feature_columns(frame)
    n_discordant = len(frame)
    n_crashes = frame.groupby(["year", "st_case"]).ngroups if n_discordant else 0
    empty = CondLogit(feature_cols=tuple(all_cols), coef={}, se={}, n_pairs=n_pairs,
                      n_discordant_pairs=n_discordant, n_crashes=n_crashes, converged=False)
    if n_discordant == 0:
        return empty

    X_full = frame[all_cols].to_numpy(dtype=float)
    live = ~np.all(X_full == 0, axis=0)
    fit_cols = [c for c, keep in zip(all_cols, live) if keep]
    dropped_cols = [c for c, keep in zip(all_cols, live) if not keep]
    if not fit_cols or n_discordant <= len(fit_cols) or frame["y"].nunique() < 2:
        return empty

    X = frame[fit_cols].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    groups = (frame["year"].astype(str) + "_" + frame["st_case"].astype(str)).to_numpy()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError):
        return empty
    if not result.mle_retvals.get("converged", True):
        return empty

    coef = dict.fromkeys(dropped_cols, 0.0)
    se: dict[str, float] = dict.fromkeys(dropped_cols, float("nan"))
    coef.update(zip(fit_cols, (float(b) for b in result.params)))
    se.update(zip(fit_cols, (float(s) for s in result.bse)))
    return CondLogit(feature_cols=tuple(all_cols), coef=coef, se=se, n_pairs=n_pairs,
                     n_discordant_pairs=n_discordant, n_crashes=n_crashes, converged=True)


def condlogit_bootstrap_ci(frame: pd.DataFrame, name: str, reps: int = 500,
                           seed: int = 20260901) -> tuple[float | None, float | None]:
    """Percentile CI for one coefficient's odds ratio, resampling whole
    crashes and refitting per replicate - the method-consistency cross-check
    against `fit_condlogit`'s cluster-robust analytic CI (design doc
    section 2), not the primary CI for any headline estimand."""
    feature_cols = feature_columns(frame)
    if frame.empty or name not in feature_cols:
        return None, None
    crash_id = (frame["year"].astype(str) + "_" + frame["st_case"].astype(str)).to_numpy()
    unique_crashes = np.unique(crash_id)
    idx_by_crash = {c: np.where(crash_id == c)[0] for c in unique_crashes}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        chosen = rng.choice(unique_crashes, size=len(unique_crashes), replace=True)
        rows = np.concatenate([idx_by_crash[c] for c in chosen])
        sub = frame.iloc[rows]
        fit = fit_condlogit(sub, n_pairs=len(sub))
        or_ = fit.odds_ratio(name)
        if or_ is not None:
            draws.append(or_)
    if not draws:
        return None, None
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def condlogit_pooled(occ: pd.DataFrame, age_form: str = "quadratic") -> CondLogit:
    """Single band-free female coefficient, all bands pooled. Validation
    target only: with no age variation this must equal v0's
    decompose().sex_effect (test_pooled_matches_v0_decompose)."""
    paired = _pivot_pairs(occ)
    frame = condlogit_frame(occ, band_mode="pooled", age_form=age_form)
    return fit_condlogit(frame, n_pairs=len(paired))


def condlogit_by_band(occ: pd.DataFrame, age_form: str = "quadratic") -> CondLogit:
    """The default trend model: one fit, one female coefficient per
    model-year band actually present, beta_seat/age SHARED (pooled) across
    all of them."""
    paired = _pivot_pairs(occ)
    frame = condlogit_frame(occ, band_mode="bands", age_form=age_form)
    return fit_condlogit(frame, n_pairs=len(paired))


def condlogit_by_band_separate(occ: pd.DataFrame,
                               age_form: str = "quadratic") -> dict[str, CondLogit]:
    """Sensitivity check: refit beta_seat/age independently inside each band
    (never pooled), bounding how much the default model's trend answer
    depends on cross-band homogeneity of those nuisance parameters."""
    out = {}
    for lo, hi in MOD_YEAR_BANDS:
        label = f"{lo}-{hi}"
        band_occ = occ[(occ["mod_year"] >= lo) & (occ["mod_year"] <= hi)]
        paired = _pivot_pairs(band_occ)
        frame = condlogit_frame(band_occ, band_mode="bands", age_form=age_form)
        out[label] = fit_condlogit(frame, n_pairs=len(paired))
    return out


# With calendar years 2000-2024 pooled, a model-year band is observed at very
# different vehicle ages across bands: a 1995 vehicle enters this data mostly
# at age 5-25, a 2020 vehicle only at age 0-4. Vehicle age at crash (who still
# drives a 20-year-old car, how it is maintained, where it crashes) is a
# between-band confounder of the female-effect trend that the within-vehicle
# design cannot difference away - it is constant within a pair. This cap
# restricts every band to its first VEHICLE_AGE_MAX years on the road, putting
# the bands on comparable vehicle-age support; the price is that band and
# calendar period become nearly synonymous, so the capped fit reads as
# "model-year trend and period trend jointly", same as the published FARS
# trend estimates. Run next to the uncapped fit, never instead of it.
VEHICLE_AGE_MAX = 12


def condlogit_by_band_vehage(occ: pd.DataFrame,
                             age_form: str = "quadratic") -> tuple[CondLogit, pd.DataFrame]:
    """The default band model refit on vehicles at most VEHICLE_AGE_MAX years
    old at crash. Returns the fit plus the capped occupant frame (the caller
    needs the latter for per-band counts). No lower bound: mod_year one year
    ahead of the calendar year is a normal new-model-year sale, not an error."""
    capped = occ[(occ["year"] - occ["mod_year"]) <= VEHICLE_AGE_MAX]
    paired = _pivot_pairs(capped)
    frame = condlogit_frame(capped, band_mode="bands", age_form=age_form)
    return fit_condlogit(frame, n_pairs=len(paired)), capped


def condlogit_trend(occ: pd.DataFrame, age_form: str = "quadratic") -> CondLogit:
    """Continuous model-year trend (log-odds female effect per decade),
    fit separately from the band dummies - never jointly, band being a
    coarsened function of year makes the two collinear together."""
    paired = _pivot_pairs(occ)
    frame = condlogit_frame(occ, band_mode="trend", age_form=age_form)
    return fit_condlogit(frame, n_pairs=len(paired))


def _band_counts(paired_all: pd.DataFrame) -> dict[str, dict[str, int]]:
    """label -> {n_pairs, n_discordant_pairs, n_crashes} for each model-year
    band, n_crashes counted over the eligible (not just discordant) pairs to
    match DoublePair's convention."""
    if paired_all.empty:
        return {f"{lo}-{hi}": {"n_pairs": 0, "n_discordant_pairs": 0, "n_crashes": 0}
               for lo, hi in MOD_YEAR_BANDS}
    labels = _band_labels(paired_all["mod_year_f"].to_numpy())
    out = {}
    for lo, hi in MOD_YEAR_BANDS:
        label = f"{lo}-{hi}"
        sub = paired_all[labels == label]
        disc = sub[(sub.died_f + sub.died_m) == 1]
        out[label] = {
            "n_pairs": len(sub), "n_discordant_pairs": len(disc),
            "n_crashes": sub.groupby(["year", "st_case"]).ngroups if len(sub) else 0,
        }
    return out


# --------------------------------------------------------------------------
# Same-sex channel: the seat x sex interaction test (design doc section 6).
#
# Within a mixed-sex pair her indicator minus his is identically 1, so a
# female x seat interaction term is algebraically identical to the seat main
# effect - not underpowered, unidentified. This channel sidesteps that by
# computing a driver-vs-passenger fatality OR separately within male-male and
# female-female discordant pairs, where the sex contrast is structurally
# absent. Never pooled with the mixed-pair channel's beta_seat/age, and never
# used to fit them - see test_no_cross_channel_leakage_from_samesex_pairs.
# --------------------------------------------------------------------------

SAMESEX_SQL = f"""
WITH occ AS (
    SELECT year, st_case, veh_no, is_female, died, seat_pos, age
    FROM v_occupant
    WHERE source = :source AND year BETWEEN :y0 AND :y1
      AND is_occupant = 1 AND is_front_outboard = 1
      AND is_belted = 1 AND {{frontal}} AND is_light_vehicle = 1
      AND age_in_range = 1 AND is_female IS NOT NULL AND died IS NOT NULL
),
pairs AS (
    -- exactly one driver, one right-front passenger, both the same sex
    SELECT year, st_case, veh_no,
           MAX(CASE WHEN seat_pos = {cb.SEAT_POS_DRIVER} THEN died END) AS drv_died,
           MAX(CASE WHEN seat_pos = {cb.SEAT_POS_RIGHT_FRONT} THEN died END) AS pax_died,
           MAX(CASE WHEN seat_pos = {cb.SEAT_POS_DRIVER} THEN age END) AS drv_age,
           MAX(CASE WHEN seat_pos = {cb.SEAT_POS_RIGHT_FRONT} THEN age END) AS pax_age,
           SUM(is_female) AS n_female
    FROM occ
    GROUP BY year, st_case, veh_no
    HAVING SUM(CASE WHEN seat_pos = {cb.SEAT_POS_DRIVER} THEN 1 ELSE 0 END) = 1
       AND SUM(CASE WHEN seat_pos = {cb.SEAT_POS_RIGHT_FRONT} THEN 1 ELSE 0 END) = 1
       AND SUM(is_female) IN (0, 2)
)
SELECT year, st_case, veh_no, drv_died, pax_died, drv_age, pax_age, n_female FROM pairs
"""


def samesex_pairs_frame(conn: sqlite3.Connection, years: tuple[int, int],
                        frontal: str = "core", sex: str = "male",
                        source: str = "fars") -> pd.DataFrame:
    """Same occupant eligibility as `occupant_frame`, restricted to vehicles
    with exactly two eligible occupants of the SAME sex, one driver and one
    right-front passenger."""
    target = {"male": 0, "female": 2}[sex]
    sql = SAMESEX_SQL.format(frontal=FRONTAL_PREDICATES[frontal])
    df = pd.read_sql_query(
        sql, conn, params={"source": source, "y0": years[0], "y1": years[1]})
    return df[df.n_female == target].reset_index(drop=True)


def _samesex_indicators(frame: pd.DataFrame) -> dict[str, pd.Series]:
    pax_only = (frame.pax_died == 1) & (frame.drv_died == 0)
    drv_only = (frame.drv_died == 1) & (frame.pax_died == 0)
    return {"pax_only": pax_only.astype(int), "drv_only": drv_only.astype(int)}


@dataclass
class SamesexSeat:
    sex: str
    frontal: str
    years: tuple[int, int]
    pax_only: int          # right-front died, driver survived
    drv_only: int          # driver died, right-front survived
    n_pairs: int
    n_crashes: int
    rr: float | None       # pax_only / drv_only - right-front fatality OR vs driving
    ci_lo: float | None
    ci_hi: float | None


def samesex_seat_effect(frame: pd.DataFrame, sex: str = "male", frontal: str = "core",
                        years: tuple[int, int] = (0, 0), reps: int = 2000,
                        seed: int = 20260901) -> SamesexSeat:
    """Driver-vs-passenger fatality OR within one same-sex cohort - the sex
    contrast is structurally absent, so this is a physical seat baseline, not
    a female effect. Same discordant-pair / crash-bootstrap machinery as
    `double_pair`."""
    ind = _samesex_indicators(frame)
    pax_only, drv_only = int(ind["pax_only"].sum()), int(ind["drv_only"].sum())
    rr = round(pax_only / drv_only, 3) if drv_only > 0 else None
    ci_lo = ci_hi = None
    matrix = _per_crash(frame, ind)
    if len(matrix):
        ratios = [p / d for p, d in _bootstrap_crashes(matrix, reps, seed) if d > 0]
        if ratios:
            lo, hi = np.percentile(ratios, [2.5, 97.5])
            ci_lo, ci_hi = round(float(lo), 3), round(float(hi), 3)
    return SamesexSeat(
        sex=sex, frontal=frontal, years=years, pax_only=pax_only, drv_only=drv_only,
        n_pairs=len(frame),
        n_crashes=frame.groupby(["year", "st_case"]).ngroups if len(frame) else 0,
        rr=rr, ci_lo=ci_lo, ci_hi=ci_hi,
    )


@dataclass
class SeatSexInteraction:
    frontal: str
    years: tuple[int, int]
    male_rr: float | None
    female_rr: float | None
    log_ratio: float | None       # log(female_rr / male_rr) - the interaction itself
    ci_lo: float | None
    ci_hi: float | None
    n_pairs_male: int
    n_pairs_female: int
    n_crashes_male: int
    n_crashes_female: int


def seatsex_interaction(male_frame: pd.DataFrame, female_frame: pd.DataFrame,
                        frontal: str = "core", years: tuple[int, int] = (0, 0),
                        reps: int = 2000, seed: int = 20260901) -> SeatSexInteraction:
    """Log-ratio of the two same-sex seat baselines - the identified seat x
    sex interaction test (design doc section 6). The two cohorts are
    resampled INDEPENDENTLY; a crash can in principle contribute one vehicle
    to each cohort, but such crashes are ~1% of either cohort on real FARS,
    so the independence approximation is negligible (unlike decompose()'s
    two seat strata, which genuinely share crashes and are resampled
    together inside one replicate). NOTE: this is the UNADJUSTED contrast -
    the two cohorts differ sharply in within-pair age structure on real
    data (female-female pairs carry a much older passenger far more often),
    so read it only next to seatsex_interaction_ageadj."""
    male = samesex_seat_effect(male_frame, sex="male", frontal=frontal, years=years, reps=0)
    female = samesex_seat_effect(female_frame, sex="female", frontal=frontal, years=years, reps=0)
    log_ratio = float(np.log(female.rr / male.rr)) if male.rr and female.rr else None

    m_matrix = _per_crash(male_frame, _samesex_indicators(male_frame))
    f_matrix = _per_crash(female_frame, _samesex_indicators(female_frame))
    draws = []
    if len(m_matrix) and len(f_matrix):
        m_draws = _bootstrap_crashes(m_matrix, reps, seed)
        f_draws = _bootstrap_crashes(f_matrix, reps, seed + 1)
        for (m_pax, m_drv), (f_pax, f_drv) in zip(m_draws, f_draws):
            if m_drv > 0 and f_drv > 0 and m_pax > 0 and f_pax > 0:
                draws.append(np.log((f_pax / f_drv) / (m_pax / m_drv)))
    ci_lo = ci_hi = None
    if draws:
        lo, hi = np.percentile(draws, [2.5, 97.5])
        ci_lo, ci_hi = round(float(lo), 4), round(float(hi), 4)

    return SeatSexInteraction(
        frontal=frontal, years=years, male_rr=male.rr, female_rr=female.rr,
        log_ratio=round(log_ratio, 4) if log_ratio is not None else None,
        ci_lo=ci_lo, ci_hi=ci_hi,
        n_pairs_male=male.n_pairs, n_pairs_female=female.n_pairs,
        n_crashes_male=male.n_crashes, n_crashes_female=female.n_crashes,
    )


# The age-adjusted interaction leans on the quadratic age model most heavily
# in exactly the large-age-gap (plausibly parent-child) pairs that dominate
# what remains of the effect after adjustment. This cap defines the
# "age-comparable" diagnostic subset: pairs whose within-pair age gap is small
# enough that the adjustment interpolates rather than extrapolates. The
# interaction refit on this subset is the fragility check for the full-cohort
# ageadj row - if the two disagree, the difference is carried by the pairs
# where the age model is trusted furthest.
AGE_COMPARABLE_GAP = 10


def age_comparable(frame: pd.DataFrame,
                   max_gap: int = AGE_COMPARABLE_GAP) -> pd.DataFrame:
    """Same-sex pairs whose |passenger age - driver age| <= max_gap."""
    if frame.empty:
        return frame
    gap = (frame["pax_age"] - frame["drv_age"]).abs()
    return frame[gap <= max_gap].reset_index(drop=True)


def samesex_condlogit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Same-sex pairs -> a fit-ready differenced frame for the AGE-ADJUSTED
    seat baseline: y = 1 iff the right-front PASSENGER died, x_const carries
    the passenger-vs-driver seat effect, and the quadratic age terms are the
    passenger's value minus the driver's - the same closed-form reduction the
    mixed channel uses, pointed at the seat contrast instead of the sex one.
    Needed because the two same-sex cohorts have very different within-pair
    age structure on real data, which loads directly onto the unadjusted
    seat ORs and hence onto their log-ratio."""
    disc = frame[(frame.drv_died + frame.pax_died) == 1]
    if disc.empty:
        return pd.DataFrame(columns=["year", "st_case", "y", "x_const", "x_age", "x_age_curv"])
    x_age, x_age_curv = age_diff_quadratic(
        disc["pax_age"].to_numpy(dtype=float), disc["drv_age"].to_numpy(dtype=float))
    return pd.DataFrame({
        "year": disc["year"].to_numpy(), "st_case": disc["st_case"].to_numpy(),
        "y": disc["pax_died"].astype(int).to_numpy(), "x_const": 1.0,
        "x_age": x_age, "x_age_curv": x_age_curv,
    }).reset_index(drop=True)


def samesex_seat_effect_ageadj(frame: pd.DataFrame) -> CondLogit:
    """Age-adjusted passenger-vs-driver fatality OR within one same-sex
    cohort: exp(coef of x_const) from the differenced logit above, with the
    same cluster-robust SEs on (year, st_case) as every condlogit fit."""
    return fit_condlogit(samesex_condlogit_frame(frame), n_pairs=len(frame))


def seatsex_interaction_ageadj(male_frame: pd.DataFrame,
                               female_frame: pd.DataFrame) -> SeatSexInteraction | None:
    """Age-adjusted seat x sex interaction: the difference of the two
    cohorts' x_const log-coefficients, with a normal CI from the combined
    cluster-robust SEs. The two fits are treated as independent - the
    cohorts overlap in ~1% of crashes on real FARS, a negligible
    misspecification (see seatsex_interaction). Returns None when either
    cohort's fit is empty or unconverged."""
    male = samesex_seat_effect_ageadj(male_frame)
    female = samesex_seat_effect_ageadj(female_frame)
    b_m, b_f = male.coef.get("x_const"), female.coef.get("x_const")
    s_m, s_f = male.se.get("x_const"), female.se.get("x_const")
    if b_m is None or b_f is None or s_m is None or s_f is None:
        return None
    if not (np.isfinite(s_m) and np.isfinite(s_f)):
        return None
    diff = b_f - b_m
    se = float(np.sqrt(s_m ** 2 + s_f ** 2))
    z = 1.959963985
    return SeatSexInteraction(
        frontal="", years=(0, 0),
        male_rr=male.odds_ratio("x_const"), female_rr=female.odds_ratio("x_const"),
        log_ratio=round(float(diff), 4),
        ci_lo=round(float(diff - z * se), 4), ci_hi=round(float(diff + z * se), 4),
        n_pairs_male=male.n_pairs, n_pairs_female=female.n_pairs,
        n_crashes_male=male.n_crashes, n_crashes_female=female.n_crashes,
    )


# --------------------------------------------------------------------------
# Power floor, v1 results writer, and the v1 run entry point.
# --------------------------------------------------------------------------

POWER_FLOOR = 30  # discordant pairs; below this a point estimate is not headlined


def headline_safe_mask(results: pd.DataFrame) -> pd.Series:
    """True where a results row carries >= POWER_FLOOR discordant pairs, read
    from its serialized cohort_def. v0 rows carry no discordant/concordant
    split at all and are always headline-safe."""
    def ok(cohort_def: str) -> bool:
        n = json.loads(cohort_def).get("n_discordant_pairs")
        return n is None or n >= POWER_FLOOR
    return results["cohort_def"].apply(ok)


def filter_headline_safe(results: pd.DataFrame) -> pd.DataFrame:
    """Rows the dashboard may render at full visual weight. Never drops rows
    from the underlying ledger - `results` itself stays the full audit
    trail; this is only what to highlight."""
    return results[headline_safe_mask(results)].reset_index(drop=True)


def _cohort_def_v1(frontal: str, n_pairs: int, n_discordant_pairs: int) -> str:
    """v0's cohort_def plus the eligible/discordant split - a discordant-only
    fit is correct for the odds ratio, but a bare n_pairs that doesn't
    distinguish the two overstates effective power."""
    info = json.loads(cb.cohort_def(frontal))
    info["n_pairs"] = n_pairs
    info["n_discordant_pairs"] = n_discordant_pairs
    return json.dumps(info, sort_keys=True)


def write_result_v1(conn: sqlite3.Connection, run_ts: str, estimand: str,
                    years: tuple[int, int], frontal: str,
                    point: float | None, ci: tuple[float | None, float | None],
                    n_pairs: int, n_discordant_pairs: int, n_crashes: int) -> None:
    """Same `results` row shape as `write_result`, but the cohort_def carries
    n_discordant_pairs alongside n_pairs - v0's write_result always derives
    cohort_def from `frontal` alone and has no room for that second count, so
    it is not reused verbatim for v1 rows (every column stays the same)."""
    conn.execute(
        """INSERT OR REPLACE INTO results
           (run_ts, git_commit, estimand, fars_years, cohort_def,
            point, ci_lo, ci_hi, n_pairs, n_crashes, n_persons)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_ts, git_commit(), estimand,
         f"{years[0]}-{years[1]}", _cohort_def_v1(frontal, n_pairs, n_discordant_pairs),
         point, ci[0], ci[1], n_pairs, n_crashes, 2 * n_pairs),
    )
    conn.commit()


def analyze_v1(conn: sqlite3.Connection, years: tuple[int, int], reps: int = 2000) -> str:
    """Every v1 estimand (design doc section 4), all frontal variants, under
    one run_ts. Additive only - never touches the fars_doublepair_* rows
    analyze() writes. Returns the run_ts written.

    Count convention, uniform across every row: n_pairs and n_crashes count
    ELIGIBLE pairs/crashes (DoublePair's convention); n_discordant_pairs in
    cohort_def counts the informative rows the fit actually used."""
    run_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for frontal in ("core", "wide", "headon"):
        occ = occupant_frame(conn, years, frontal)
        paired_all = _pivot_pairs(occ)
        band_info = _band_counts(paired_all)
        crashes_all = paired_all.groupby(["year", "st_case"]).ngroups if len(paired_all) else 0

        band_fit = condlogit_by_band(occ)
        for lo, hi in MOD_YEAR_BANDS:
            label = f"{lo}-{hi}"
            col = f"x_female_band_{label}"
            info = band_info[label]
            write_result_v1(
                conn, run_ts, f"fars_condlogit_sex_effect_frontal_{frontal}_band_{label}",
                years, frontal, band_fit.odds_ratio(col), band_fit.or_ci(col),
                info["n_pairs"], info["n_discordant_pairs"], info["n_crashes"])

        write_result_v1(
            conn, run_ts, f"fars_condlogit_seat_effect_rightfront_frontal_{frontal}",
            years, frontal, band_fit.odds_ratio("x_seat"), band_fit.or_ci("x_seat"),
            len(paired_all), band_fit.n_discordant_pairs, crashes_all)
        write_result_v1(
            conn, run_ts, f"fars_condlogit_age_slope_frontal_{frontal}",
            years, frontal, band_fit.odds_ratio("x_age"), band_fit.or_ci("x_age"),
            len(paired_all), band_fit.n_discordant_pairs, crashes_all)
        write_result_v1(
            conn, run_ts, f"fars_condlogit_age_curvature_frontal_{frontal}",
            years, frontal, band_fit.odds_ratio("x_age_curv"), band_fit.or_ci("x_age_curv"),
            len(paired_all), band_fit.n_discordant_pairs, crashes_all)

        pooled_fit = condlogit_pooled(occ)
        write_result_v1(
            conn, run_ts, f"fars_condlogit_sex_effect_frontal_{frontal}_pooled",
            years, frontal, pooled_fit.odds_ratio("x_const"), pooled_fit.or_ci("x_const"),
            pooled_fit.n_pairs, pooled_fit.n_discordant_pairs, crashes_all)

        for label, fit in condlogit_by_band_separate(occ).items():
            col = f"x_female_band_{label}"
            info = band_info[label]
            write_result_v1(
                conn, run_ts,
                f"fars_condlogit_sex_effect_frontal_{frontal}_band_{label}_separatenuisance",
                years, frontal, fit.odds_ratio(col), fit.or_ci(col),
                info["n_pairs"], info["n_discordant_pairs"], info["n_crashes"])

        pw_fit = condlogit_by_band(occ, age_form="piecewise")
        for lo, hi in MOD_YEAR_BANDS:
            label = f"{lo}-{hi}"
            col = f"x_female_band_{label}"
            info = band_info[label]
            write_result_v1(
                conn, run_ts,
                f"fars_condlogit_sex_effect_frontal_{frontal}_band_{label}_agepiecewise",
                years, frontal, pw_fit.odds_ratio(col), pw_fit.or_ci(col),
                info["n_pairs"], info["n_discordant_pairs"], info["n_crashes"])

        vehage_fit, vehage_occ = condlogit_by_band_vehage(occ)
        vehage_info = _band_counts(_pivot_pairs(vehage_occ))
        for lo, hi in MOD_YEAR_BANDS:
            label = f"{lo}-{hi}"
            col = f"x_female_band_{label}"
            info = vehage_info[label]
            write_result_v1(
                conn, run_ts,
                f"fars_condlogit_sex_effect_frontal_{frontal}_band_{label}_vehage{VEHICLE_AGE_MAX}",
                years, frontal, vehage_fit.odds_ratio(col), vehage_fit.or_ci(col),
                info["n_pairs"], info["n_discordant_pairs"], info["n_crashes"])

        trend_fit = condlogit_trend(occ)
        trend_point = trend_fit.coef.get("x_trend")
        write_result_v1(
            conn, run_ts, f"fars_condlogit_sex_trend_slope_frontal_{frontal}",
            years, frontal, round(trend_point, 6) if trend_point is not None else None,
            trend_fit.coef_ci("x_trend"),
            trend_fit.n_pairs, trend_fit.n_discordant_pairs, crashes_all)

        male_frame = samesex_pairs_frame(conn, years, frontal, sex="male")
        female_frame = samesex_pairs_frame(conn, years, frontal, sex="female")
        male_seat = samesex_seat_effect(male_frame, sex="male", frontal=frontal,
                                        years=years, reps=reps)
        female_seat = samesex_seat_effect(female_frame, sex="female", frontal=frontal,
                                          years=years, reps=reps)
        write_result_v1(
            conn, run_ts, f"fars_samesex_seat_effect_male_frontal_{frontal}",
            years, frontal, male_seat.rr, (male_seat.ci_lo, male_seat.ci_hi),
            male_seat.n_pairs, male_seat.pax_only + male_seat.drv_only, male_seat.n_crashes)
        write_result_v1(
            conn, run_ts, f"fars_samesex_seat_effect_female_frontal_{frontal}",
            years, frontal, female_seat.rr, (female_seat.ci_lo, female_seat.ci_hi),
            female_seat.n_pairs, female_seat.pax_only + female_seat.drv_only, female_seat.n_crashes)

        interaction = seatsex_interaction(male_frame, female_frame, frontal=frontal,
                                          years=years, reps=reps)
        write_result_v1(
            conn, run_ts, f"fars_samesex_seatsex_interaction_frontal_{frontal}",
            years, frontal, interaction.log_ratio, (interaction.ci_lo, interaction.ci_hi),
            interaction.n_pairs_male + interaction.n_pairs_female,
            (male_seat.pax_only + male_seat.drv_only) + (female_seat.pax_only + female_seat.drv_only),
            interaction.n_crashes_male + interaction.n_crashes_female)

        # Age-adjusted same-sex rows: the two cohorts differ sharply in
        # within-pair age structure, and that composition loads directly onto
        # the unadjusted ORs above - these are the headline versions.
        male_crashes = male_frame.groupby(["year", "st_case"]).ngroups if len(male_frame) else 0
        female_crashes = female_frame.groupby(["year", "st_case"]).ngroups if len(female_frame) else 0
        male_adj = samesex_seat_effect_ageadj(male_frame)
        female_adj = samesex_seat_effect_ageadj(female_frame)
        write_result_v1(
            conn, run_ts, f"fars_samesex_seat_effect_male_ageadj_frontal_{frontal}",
            years, frontal, male_adj.odds_ratio("x_const"), male_adj.or_ci("x_const"),
            len(male_frame), male_adj.n_discordant_pairs, male_crashes)
        write_result_v1(
            conn, run_ts, f"fars_samesex_seat_effect_female_ageadj_frontal_{frontal}",
            years, frontal, female_adj.odds_ratio("x_const"), female_adj.or_ci("x_const"),
            len(female_frame), female_adj.n_discordant_pairs, female_crashes)
        interaction_adj = seatsex_interaction_ageadj(male_frame, female_frame)
        write_result_v1(
            conn, run_ts, f"fars_samesex_seatsex_interaction_ageadj_frontal_{frontal}",
            years, frontal,
            interaction_adj.log_ratio if interaction_adj else None,
            (interaction_adj.ci_lo, interaction_adj.ci_hi) if interaction_adj else (None, None),
            len(male_frame) + len(female_frame),
            male_adj.n_discordant_pairs + female_adj.n_discordant_pairs,
            male_crashes + female_crashes)

        # Fragility check for the row above: the same age-adjusted interaction
        # refit on age-comparable pairs only, where the quadratic age model
        # interpolates instead of extrapolating.
        male_cmp, female_cmp = age_comparable(male_frame), age_comparable(female_frame)
        male_cmp_adj = samesex_seat_effect_ageadj(male_cmp)
        female_cmp_adj = samesex_seat_effect_ageadj(female_cmp)
        interaction_cmp = seatsex_interaction_ageadj(male_cmp, female_cmp)
        cmp_crashes = (
            (male_cmp.groupby(["year", "st_case"]).ngroups if len(male_cmp) else 0)
            + (female_cmp.groupby(["year", "st_case"]).ngroups if len(female_cmp) else 0))
        write_result_v1(
            conn, run_ts,
            f"fars_samesex_seatsex_interaction_ageadj_agecomparable_frontal_{frontal}",
            years, frontal,
            interaction_cmp.log_ratio if interaction_cmp else None,
            (interaction_cmp.ci_lo, interaction_cmp.ci_hi) if interaction_cmp else (None, None),
            len(male_cmp) + len(female_cmp),
            male_cmp_adj.n_discordant_pairs + female_cmp_adj.n_discordant_pairs,
            cmp_crashes)
    return run_ts
