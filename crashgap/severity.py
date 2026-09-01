"""v2: design-weighted severity logistic on CISS / NASS-CDS (docs/v2-design.md).

The estimator is the survey convention end to end: a weighted logistic
pseudo-MLE with Taylor-linearized variance for a stratified,
with-replacement, single-stage-cluster design (strata = PSUSTRAT, clusters =
PSU, weights = CASEWGT / RATWGT), and t-based CIs on df = n_PSU - n_strata.
With ~32 PSUs in 12 strata (CISS) that df is 20, so the t quantile is
materially wider than a normal one - using z here would overstate certainty
on exactly the rung whose whole point is honest severity-adjusted intervals.

Years are pooled within a source: PSUs are persistent physical sites, so
same-PSU rows from different years belong to ONE cluster and (PSUSTRAT, PSU)
carry over unchanged. NASS and CISS are never pooled with each other
(different designs, eras, AIS revisions).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from . import codebook as cb
from . import severity_codebook as scb
from .analysis import MOD_YEAR_BANDS, git_commit

# Covariate centering/scale constants - cosmetic conditioning only, pinned so
# coefficients are reproducible run to run.
AGE_CENTER, AGE_SCALE = 45.0, 10.0
DV_CENTER, DV_SCALE = 25.0, 10.0
HEIGHT_CENTER, HEIGHT_SCALE = 170.0, 10.0
BMI_CENTER, BMI_SCALE = 27.0, 5.0

WEIGHT_TRIM_PCT = 95  # the _wtrim95 sensitivity caps weights at this percentile


@dataclass
class SvyLogit:
    """One design-weighted logistic fit. coef/se are log-odds; the CI is
    t-based on df = n_clusters - n_strata (None SE when the fit failed)."""
    feature_cols: tuple[str, ...]
    coef: dict[str, float]
    se: dict[str, float]
    df: int
    n_obs: int
    n_cases: int
    n_psus: int
    n_strata: int
    n_events: int
    sum_weight: float
    kish_neff: float
    weight_max_over_median: float
    converged: bool

    def odds_ratio(self, name: str) -> float | None:
        b = self.coef.get(name)
        return round(float(np.exp(b)), 4) if b is not None else None

    def or_ci(self, name: str) -> tuple[float | None, float | None]:
        b, s = self.coef.get(name), self.se.get(name)
        if b is None or s is None or not np.isfinite(s) or self.df < 1:
            return None, None
        t = float(stats.t.ppf(0.975, self.df))
        return (round(float(np.exp(b - t * s)), 4),
                round(float(np.exp(b + t * s)), 4))


def _cluster_scores(scores: np.ndarray, strata: np.ndarray,
                    clusters: np.ndarray) -> np.ndarray:
    """Taylor linearization B: sum over strata of n_h/(n_h-1) times the
    centered outer products of per-cluster score totals. A singleton stratum
    centers against the grand mean of all cluster totals instead of crashing
    (the svy 'adjust' convention)."""
    frame = pd.DataFrame(scores)
    frame["_stratum"] = strata
    frame["_cluster"] = clusters
    totals = frame.groupby(["_stratum", "_cluster"]).sum()
    k = scores.shape[1]
    grand_mean = totals.to_numpy().mean(axis=0)
    b = np.zeros((k, k))
    for _, z_h in totals.groupby(level="_stratum"):
        z = z_h.to_numpy()
        n_h = len(z)
        if n_h == 1:
            dev = z[0] - grand_mean
            b += np.outer(dev, dev)
            continue
        dev = z - z.mean(axis=0)
        b += (n_h / (n_h - 1)) * dev.T @ dev
    return b


def fit_svylogit(frame: pd.DataFrame, feature_cols: list[str]) -> SvyLogit:
    """frame needs: y (0/1), w (>0), psustrat, psu, case_id, and the
    feature columns. An intercept is added here; a feature with zero
    variance is dropped from the fit but still reported (coef 0, se NaN),
    mirroring fit_condlogit's contract."""
    n_obs = len(frame)
    w = frame["w"].to_numpy(dtype=float)
    strata = frame["psustrat"].to_numpy()
    clusters = frame["psu"].to_numpy()
    n_psus = len(pd.unique(pd.Series(list(zip(strata, clusters)))))
    n_strata = len(np.unique(strata)) if n_obs else 0
    diag = {
        "n_obs": n_obs,
        "n_cases": frame.groupby(["psu", "case_id"]).ngroups if n_obs else 0,
        "n_psus": n_psus, "n_strata": n_strata,
        "n_events": int(frame["y"].sum()) if n_obs else 0,
        "sum_weight": round(float(w.sum()), 1) if n_obs else 0.0,
        "kish_neff": round(float(w.sum() ** 2 / (w ** 2).sum()), 1) if n_obs else 0.0,
        "weight_max_over_median": (
            round(float(w.max() / np.median(w)), 1) if n_obs else 0.0),
    }
    df = n_psus - n_strata
    empty = SvyLogit(feature_cols=tuple(feature_cols), coef={}, se={}, df=df,
                     converged=False, **diag)
    if n_obs == 0 or frame["y"].nunique() < 2:
        return empty

    live = [c for c in feature_cols if frame[c].to_numpy(dtype=float).std() > 0]
    dropped = [c for c in feature_cols if c not in live]
    if not live or df < 1:
        return empty

    X = sm.add_constant(frame[live].to_numpy(dtype=float), has_constant="add")
    y = frame["y"].to_numpy(dtype=float)
    try:
        result = sm.GLM(y, X, family=sm.families.Binomial(),
                        freq_weights=w).fit(maxiter=200)
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError):
        return empty
    if not result.converged:
        return empty

    p = np.asarray(result.predict())
    # A: weighted information; B: linearized between-cluster variability.
    a = X.T @ (X * (w * p * (1 - p))[:, None])
    scores = X * (w * (y - p))[:, None]
    b = _cluster_scores(scores, strata, clusters)
    try:
        a_inv = np.linalg.inv(a)
    except np.linalg.LinAlgError:
        return empty
    v = a_inv @ b @ a_inv
    names = ["const"] + live
    coef = dict.fromkeys(dropped, 0.0)
    se: dict[str, float] = dict.fromkeys(dropped, float("nan"))
    coef.update(zip(names, (float(x) for x in result.params)))
    se.update(zip(names, (float(x) for x in np.sqrt(np.clip(np.diag(v), 0, None)))))
    return SvyLogit(feature_cols=tuple(feature_cols), coef=coef, se=se, df=df,
                    converged=True, **diag)


# --------------------------------------------------------------------------
# Cohort and design matrix
# --------------------------------------------------------------------------

SEV_COHORT_SQL = f"""
SELECT psu, psustrat, case_id, veh_no, occ_no, weight AS w, sex_female, age,
       seat_pos, height, bmi, mais, mais08, mod_year, dv_total
FROM sev_occupant
WHERE source = :source
  AND is_frontal = 1
  AND seat_pos IN ({cb.sql_in(scb.SEV_SEAT_FRONT_OUTBOARD)})
  AND belted = 1
  AND sex_female IS NOT NULL
  AND age BETWEEN {scb.SEV_AGE_MIN} AND {scb.SEV_AGE_MAX}
  AND mais IS NOT NULL
  AND weight IS NOT NULL AND weight > 0
  AND psustrat IS NOT NULL
  AND body_typ IN ({cb.sql_in(cb.BODY_TYP_LIGHT_VEHICLE)})
  AND mod_year BETWEEN {MOD_YEAR_BANDS[0][0]} AND {MOD_YEAR_BANDS[-1][1]}
"""


def severity_cohort(conn: sqlite3.Connection, source: str) -> pd.DataFrame:
    """The frontal severity cohort for one source: front-outboard belted
    adults with known sex and graded MAIS in light vehicles. Complete-case
    on the base covariates by construction of the WHERE clause; the dv and
    anthro tiers subset further in build_design."""
    return pd.read_sql_query(SEV_COHORT_SQL, conn, params={"source": source})


def build_design(cohort: pd.DataFrame, outcome: str, tier: str,
                 ais: str = "contemporary") -> tuple[pd.DataFrame, list[str]]:
    """Fit-ready frame + feature list for one (outcome, tier).

    outcome: 'mais2plus' | 'mais3plus'; tier: 'base' | 'dv' | 'dvanthro';
    ais: 'contemporary' uses the mais column, 'ais08' the NASS dual-coded
    mais08 (rows without it drop). Band dummies are reference-coded against
    the oldest band present - with an intercept in the model, one dummy per
    band would be a dummy trap."""
    threshold = {"mais2plus": 2, "mais3plus": 3}[outcome]
    frame = cohort.copy()
    if ais == "ais08":
        frame = frame[frame["mais08"].notna()]
        grade = frame["mais08"]
    else:
        grade = frame["mais"]
    frame["y"] = (grade.astype(float) >= threshold).astype(int)
    frame["female"] = frame["sex_female"].astype(float)
    frame["x_age"] = (frame["age"].astype(float) - AGE_CENTER) / AGE_SCALE
    frame["x_age2"] = frame["x_age"] ** 2
    frame["seat_rf"] = (frame["seat_pos"] == 13).astype(float)
    features = ["female", "x_age", "x_age2", "seat_rf"]

    bands = sorted({label for label in _band_of(frame["mod_year"]) if label})
    band_labels = _band_of(frame["mod_year"])
    for label in bands[1:]:  # reference = oldest present band
        col = f"band_{label}"
        frame[col] = (band_labels == label).astype(float)
        features.append(col)

    if tier in ("dv", "dvanthro"):
        frame = frame[frame["dv_total"].notna()].copy()
        frame["x_dv"] = (frame["dv_total"].astype(float) - DV_CENTER) / DV_SCALE
        frame["x_dv2"] = frame["x_dv"] ** 2
        features += ["x_dv", "x_dv2"]
    if tier == "dvanthro":
        frame = frame[frame["height"].notna() & frame["bmi"].notna()].copy()
        frame["x_height"] = (frame["height"].astype(float) - HEIGHT_CENTER) / HEIGHT_SCALE
        frame["x_bmi"] = (frame["bmi"].astype(float) - BMI_CENTER) / BMI_SCALE
        features += ["x_height", "x_bmi"]
    return frame.reset_index(drop=True), features


def _band_of(mod_year: pd.Series) -> np.ndarray:
    labels = np.full(len(mod_year), None, dtype=object)
    my = mod_year.astype(float).to_numpy()
    for lo, hi in MOD_YEAR_BANDS:
        labels[(my >= lo) & (my <= hi)] = f"{lo}-{hi}"
    return labels


def trim_weights(frame: pd.DataFrame, pct: float = WEIGHT_TRIM_PCT) -> pd.DataFrame:
    """Cap w at its cohort percentile - the weight-instability sensitivity."""
    if frame.empty:
        return frame
    cap = float(np.percentile(frame["w"].to_numpy(dtype=float), pct))
    out = frame.copy()
    out["w"] = out["w"].clip(upper=cap)
    return out


# --------------------------------------------------------------------------
# The v2 run: estimand ledger rows
# --------------------------------------------------------------------------

TIERS = ("base", "dv", "dvanthro")
OUTCOMES = ("mais2plus", "mais3plus")


def _cohort_def_v2(source: str, years: str, outcome: str, tier: str, ais: str,
                   fit: SvyLogit, share_dv_known: float | None,
                   trimmed: bool) -> str:
    return json.dumps({
        "source": source, "years": years, "outcome": outcome, "tier": tier,
        "ais_revision": {"ciss": "AIS2015"}.get(source, "AIS90") if ais == "contemporary" else "AIS2008",
        "cohort": "frontal (primary damage plane F, clock 11/12/1), front outboard, "
                  f"belted [{cb.sql_in(scb.SEV_BELTED)}], age {scb.SEV_AGE_MIN}-{scb.SEV_AGE_MAX}, "
                  "known sex + graded MAIS, light vehicle, complete-case per tier",
        "design": "svy logistic pseudo-MLE, Taylor linearized, strata=PSUSTRAT, "
                  "cluster=PSU, t df = n_PSU - n_strata; years pooled within source",
        "n_obs": fit.n_obs, "n_cases": fit.n_cases, "n_psus": fit.n_psus,
        "n_strata": fit.n_strata, "df": fit.df, "n_events": fit.n_events,
        "sum_weight": fit.sum_weight, "kish_neff": fit.kish_neff,
        "weight_max_over_median": fit.weight_max_over_median,
        "share_dv_known": share_dv_known,
        "weights_trimmed_at_pct": WEIGHT_TRIM_PCT if trimmed else None,
    }, sort_keys=True)


def _write_v2(conn: sqlite3.Connection, run_ts: str, estimand: str, years: str,
              cohort_def: str, fit: SvyLogit) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO results
           (run_ts, git_commit, estimand, fars_years, cohort_def,
            point, ci_lo, ci_hi, n_pairs, n_crashes, n_persons)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_ts, git_commit(), estimand, years, cohort_def,
         fit.odds_ratio("female"), *fit.or_ci("female"),
         None, fit.n_cases, fit.n_obs),
    )
    conn.commit()


def source_years(conn: sqlite3.Connection, source: str) -> str | None:
    row = conn.execute(
        "SELECT MIN(year), MAX(year) FROM sev_occupant WHERE source = ?",
        (source,)).fetchone()
    if row is None or row[0] is None:
        return None
    return f"{row[0]}-{row[1]}"


def analyze_v2(conn: sqlite3.Connection, reps: int = 0) -> str:
    """Every v2 estimand under one run_ts: per source x outcome, the three
    covariate tiers, a trimmed-weights sensitivity on base, and for NASS the
    AIS08 dual-coding sensitivity on base. `fars_years` carries the source's
    own year span (the column name is historical). reps is accepted for CLI
    symmetry; the variance here is analytic, nothing bootstraps."""
    run_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for source in ("ciss", "nass"):
        years = source_years(conn, source)
        if years is None:
            continue
        cohort = severity_cohort(conn, source)
        share_dv = (round(float(cohort["dv_total"].notna().mean()), 3)
                    if len(cohort) else None)
        for outcome in OUTCOMES:
            for tier in TIERS:
                frame, features = build_design(cohort, outcome, tier)
                fit = fit_svylogit(frame, features)
                _write_v2(
                    conn, run_ts,
                    f"{source}_svylogit_female_or_{outcome}_frontal_{tier}",
                    years,
                    _cohort_def_v2(source, years, outcome, tier, "contemporary",
                                   fit, share_dv if tier != "base" else None,
                                   trimmed=False),
                    fit)
            # weight-trimming sensitivity, base tier
            frame, features = build_design(cohort, outcome, "base")
            fit = fit_svylogit(trim_weights(frame), features)
            _write_v2(
                conn, run_ts,
                f"{source}_svylogit_female_or_{outcome}_frontal_base_wtrim{WEIGHT_TRIM_PCT}",
                years,
                _cohort_def_v2(source, years, outcome, "base", "contemporary",
                               fit, None, trimmed=True),
                fit)
            if source == "nass":
                # AIS revision sensitivity on the dual-coded 2010+ subset
                frame, features = build_design(cohort, outcome, "base", ais="ais08")
                fit = fit_svylogit(frame, features)
                _write_v2(
                    conn, run_ts,
                    f"{source}_svylogit_female_or_{outcome}_frontal_base_ais08",
                    years,
                    _cohort_def_v2(source, years, outcome, "base", "ais08",
                                   fit, None, trimmed=False),
                    fit)
    return run_ts
