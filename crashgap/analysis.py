"""The Evans double-pair estimate and its crash-clustered bootstrap CI.

FARS is a census of fatal crashes only, so a raw pooled odds ratio is
dominated by crash-involvement selection and can point the wrong way. The
double-pair design restricts to vehicles carrying exactly one eligible male
and one eligible female, so crash severity, delta-V, vehicle and impact
direction are held physically constant within the vehicle. Among discordant
pairs (exactly one of the two died), the female:male relative fatality risk
is (female died, male survived) / (male died, female survived).

A crash can contribute more than one vehicle pair, so occupants are not
independent: the CI is a percentile bootstrap resampling whole crashes
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

PAIRS_SQL = """
WITH occ AS (
    SELECT year, st_case, veh_no, is_female, died, mod_year
    FROM v_occupant
    WHERE source = :source AND year BETWEEN :y0 AND :y1
      AND is_occupant = 1 AND is_front_outboard = 1
      AND is_belted = 1 AND {frontal} AND is_light_vehicle = 1
      AND age_in_range = 1 AND is_female IS NOT NULL AND died IS NOT NULL
),
pairs AS (
    -- vehicles holding EXACTLY one eligible female + one eligible male
    SELECT year, st_case, veh_no,
           MAX(CASE WHEN is_female = 1 THEN died END) AS f_died,
           MAX(CASE WHEN is_female = 0 THEN died END) AS m_died,
           MAX(mod_year) AS mod_year
    FROM occ
    GROUP BY year, st_case, veh_no
    HAVING SUM(is_female) = 1 AND SUM(1 - is_female) = 1
)
SELECT year, st_case, veh_no, f_died, m_died, mod_year FROM pairs
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


def cluster_bootstrap_ci(pairs: pd.DataFrame, reps: int = 2000,
                         seed: int = 20260802) -> tuple[float | None, float | None]:
    """Percentile CI for f_only/m_only, resampling whole crashes."""
    per_crash = pairs.assign(
        f_only=((pairs.f_died == 1) & (pairs.m_died == 0)).astype(int),
        m_only=((pairs.m_died == 1) & (pairs.f_died == 0)).astype(int),
    ).groupby(["year", "st_case"])[["f_only", "m_only"]].sum()
    fo = per_crash.f_only.to_numpy()
    mo = per_crash.m_only.to_numpy()
    n = len(per_crash)
    if n == 0:
        return None, None
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(reps):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        f_sum, m_sum = counts @ fo, counts @ mo
        if m_sum > 0:
            ratios.append(f_sum / m_sum)
    if not ratios:
        return None, None
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def double_pair(conn: sqlite3.Connection, years: tuple[int, int],
                frontal: str = "core", reps: int = 2000) -> DoublePair:
    pairs = pairs_frame(conn, years, frontal)
    f_only = int(((pairs.f_died == 1) & (pairs.m_died == 0)).sum())
    m_only = int(((pairs.m_died == 1) & (pairs.f_died == 0)).sum())
    rr = round(f_only / m_only, 3) if m_only > 0 else None
    ci_lo, ci_hi = cluster_bootstrap_ci(pairs, reps=reps)
    return DoublePair(
        frontal=frontal, years=years, f_only=f_only, m_only=m_only,
        n_pairs=len(pairs),
        n_crashes=pairs.groupby(["year", "st_case"]).ngroups,
        rr=rr, ci_lo=ci_lo, ci_hi=ci_hi,
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


def write_result(conn: sqlite3.Connection, run_ts: str, dp: DoublePair) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO results
           (run_ts, git_commit, estimand, fars_years, cohort_def,
            point, ci_lo, ci_hi, n_pairs, n_crashes, n_persons)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_ts, git_commit(),
         f"fars_doublepair_fatality_f_vs_m_frontal_{dp.frontal}",
         f"{dp.years[0]}-{dp.years[1]}", cb.cohort_def(dp.frontal),
         dp.rr, dp.ci_lo, dp.ci_hi, dp.n_pairs, dp.n_crashes, dp.n_persons),
    )
    conn.commit()


def analyze(conn: sqlite3.Connection, years: tuple[int, int],
            reps: int = 2000) -> list[DoublePair]:
    """All three frontal variants, written to results under one run_ts."""
    run_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = []
    for frontal in ("core", "wide", "headon"):
        dp = double_pair(conn, years, frontal, reps=reps)
        write_result(conn, run_ts, dp)
        out.append(dp)
    return out
