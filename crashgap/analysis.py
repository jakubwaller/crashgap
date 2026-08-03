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

import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

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
