"""Planted-design tests for the v2 survey-weighted logistic (severity.py).

The estimator is hand-rolled (statsmodels has no svyglm), so these tests pin
it against things that are independently true: the weighted point estimate
must be invariant to row replication vs integer weights, the linearized
variance must agree with a standard robust sandwich in the degenerate design
where they coincide, a planted stratified-cluster simulation must be
recovered, and clustering must never *shrink* uncertainty relative to the
naive iid fit on clustered data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from crashgap.severity import (
    SvyLogit,
    build_design,
    fit_svylogit,
    trim_weights,
)


def _design_frame(n: int, seed: int, female_logodds: float = 0.5,
                  n_strata: int = 4, psus_per_stratum: int = 6,
                  cluster_sd: float = 0.3) -> pd.DataFrame:
    """Stratified single-stage-cluster sample with a cluster random effect
    and a known female effect on the log-odds scale."""
    rng = np.random.default_rng(seed)
    strata = rng.integers(0, n_strata, n)
    psu_in_stratum = rng.integers(0, psus_per_stratum, n)
    psu = strata * 100 + psu_in_stratum
    cluster_effect = {p: rng.normal(0, cluster_sd) for p in np.unique(psu)}
    female = rng.integers(0, 2, n)
    x = rng.normal(0, 1, n)
    logit = -1.0 + female_logodds * female + 0.4 * x + np.array(
        [cluster_effect[p] for p in psu])
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    w = rng.uniform(0.5, 4.0, n)
    return pd.DataFrame({
        "y": y, "w": w, "psustrat": strata, "psu": psu,
        "case_id": np.arange(n).astype(str),
        "female": female.astype(float), "x": x,
    })


def test_point_estimate_and_variance_match_row_replication():
    """Integer weight w=k must equal physically replicating the row k times -
    for the POINT ESTIMATE (the defining property of the weighted estimating
    equation) and for the LINEARIZED VARIANCE (per-cluster score totals,
    A, and B are all identical under replication). The variance half is the
    tripwire for any misplaced weight power - e.g. w**2 in the scores would
    pass every equal-weights test in this file but fail here (4x vs 2x per
    replicated pair)."""
    frame = _design_frame(400, seed=1)
    frame["w"] = np.tile([1.0, 3.0], 200)
    fit = fit_svylogit(frame, ["female", "x"])

    replicated = frame.loc[frame.index.repeat(frame["w"].astype(int))].copy()
    replicated["w"] = 1.0
    fit_rep = fit_svylogit(replicated, ["female", "x"])
    assert fit.converged and fit_rep.converged
    for name in ("const", "female", "x"):
        assert fit.coef[name] == pytest.approx(fit_rep.coef[name], abs=1e-8)
        assert fit.se[name] == pytest.approx(fit_rep.se[name], rel=1e-6)
    assert fit.df == fit_rep.df  # same clusters, same strata


def test_degenerate_design_matches_hc_sandwich():
    """One stratum, every observation its own PSU, unit weights: the
    linearized variance reduces to the HC-type sandwich with an n/(n-1)
    factor, so it must track statsmodels' HC1 closely."""
    frame = _design_frame(600, seed=2)
    frame["w"] = 1.0
    frame["psustrat"] = 0
    frame["psu"] = np.arange(len(frame))
    fit = fit_svylogit(frame, ["female", "x"])
    assert fit.converged

    X = sm.add_constant(frame[["female", "x"]].to_numpy())
    ref = sm.Logit(frame["y"].to_numpy(), X).fit(disp=0, cov_type="HC1")
    for i, name in enumerate(["const", "female", "x"]):
        assert fit.coef[name] == pytest.approx(float(ref.params[i]), abs=1e-6)
        assert fit.se[name] == pytest.approx(float(ref.bse[i]), rel=0.02)


def test_planted_female_effect_recovered():
    frame = _design_frame(20000, seed=3, female_logodds=0.5)
    fit = fit_svylogit(frame, ["female", "x"])
    assert fit.converged
    assert fit.coef["female"] == pytest.approx(0.5, abs=0.12)
    lo, hi = fit.or_ci("female")
    assert lo < np.exp(0.5) < hi


def test_clustering_widens_over_naive_iid():
    """With a real cluster random effect, design-based SEs must exceed the
    naive iid MLE SEs - if they don't, the linearization is broken."""
    frame = _design_frame(4000, seed=4, cluster_sd=0.6)
    frame["w"] = 1.0
    fit = fit_svylogit(frame, ["female", "x"])
    X = sm.add_constant(frame[["female", "x"]].to_numpy())
    naive = sm.Logit(frame["y"].to_numpy(), X).fit(disp=0)
    # the cluster effect loads on the intercept; that SE must widen clearly
    assert fit.se["const"] > float(naive.bse[0]) * 1.3


def test_t_df_is_psus_minus_strata():
    frame = _design_frame(500, seed=5, n_strata=4, psus_per_stratum=6)
    fit = fit_svylogit(frame, ["female", "x"])
    n_psus = frame.groupby(["psustrat", "psu"]).ngroups
    assert fit.df == n_psus - 4
    # t-based CI must be wider than the +/-1.96 normal one
    b, s = fit.coef["female"], fit.se["female"]
    lo, hi = fit.or_ci("female")
    assert lo < np.exp(b - 1.959 * s) and hi > np.exp(b + 1.959 * s)


def test_singleton_stratum_does_not_crash():
    frame = _design_frame(300, seed=6, n_strata=2)
    lone = frame.iloc[:1].copy()
    lone["psustrat"] = 99
    lone["psu"] = 9999
    fit = fit_svylogit(pd.concat([frame, lone], ignore_index=True), ["female", "x"])
    assert fit.converged
    assert np.isfinite(fit.se["female"])


def test_empty_and_degenerate_inputs_return_empty_fit():
    empty = fit_svylogit(pd.DataFrame(
        columns=["y", "w", "psustrat", "psu", "case_id", "female"]), ["female"])
    assert not empty.converged and empty.coef == {}

    frame = _design_frame(50, seed=7)
    frame["y"] = 1  # no outcome variation
    assert not fit_svylogit(frame, ["female", "x"]).converged


def test_diagnostics_and_trimming():
    frame = _design_frame(1000, seed=8)
    frame.loc[0, "w"] = 500.0  # one mega-weight
    fit = fit_svylogit(frame, ["female", "x"])
    assert fit.weight_max_over_median > 100
    assert fit.kish_neff < fit.n_obs  # unequal weights always cost efficiency
    assert fit.n_events == int(frame["y"].sum())

    trimmed = trim_weights(frame)
    assert trimmed["w"].max() <= np.percentile(frame["w"], 95) + 1e-9
    assert len(trimmed) == len(frame)  # trimming caps, never drops


def _cohort(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    return pd.DataFrame({
        "psu": rng.integers(0, 8, n), "psustrat": rng.integers(0, 2, n),
        "case_id": np.arange(n).astype(str), "veh_no": 1, "occ_no": 1,
        "w": rng.uniform(1, 5, n), "sex_female": rng.integers(0, 2, n),
        "age": rng.integers(16, 96, n),
        "seat_pos": rng.choice([11, 13], n),
        "height": np.where(rng.random(n) < 0.7, rng.uniform(150, 200, n), np.nan),
        "bmi": np.where(rng.random(n) < 0.7, rng.uniform(18, 40, n), np.nan),
        "mais": rng.integers(0, 6, n), "mais08": np.nan,
        "mod_year": rng.choice([1995, 2005, 2012, 2020], n),
        "dv_total": np.where(rng.random(n) < 0.6, rng.uniform(5, 80, n), np.nan),
    })


def test_build_design_outcome_and_tiers():
    cohort = _cohort()
    base, feats = build_design(cohort, "mais2plus", "base")
    assert (base["y"] == (cohort["mais"] >= 2).astype(int)).all()
    assert len(base) == len(cohort)
    # reference-coded bands: oldest present band has NO dummy (dummy trap)
    band_cols = [f for f in feats if f.startswith("band_")]
    assert "band_1970-1999" not in band_cols and len(band_cols) == 3

    m3, _ = build_design(cohort, "mais3plus", "base")
    assert (m3["y"] == (cohort["mais"] >= 3).astype(int)).all()

    dv, dv_feats = build_design(cohort, "mais2plus", "dv")
    assert len(dv) == int(cohort["dv_total"].notna().sum())
    assert "x_dv" in dv_feats and "x_dv2" in dv_feats

    anthro, an_feats = build_design(cohort, "mais2plus", "dvanthro")
    assert "x_height" in an_feats and "x_bmi" in an_feats
    expected = cohort["dv_total"].notna() & cohort["height"].notna() & cohort["bmi"].notna()
    assert len(anthro) == int(expected.sum())


def test_build_design_ais08_subsets_to_dual_coded_rows():
    cohort = _cohort()
    cohort.loc[: len(cohort) // 2, "mais08"] = 3.0
    frame, _ = build_design(cohort, "mais3plus", "base", ais="ais08")
    assert len(frame) == int(cohort["mais08"].notna().sum())
    assert (frame["y"] == 1).all()  # every dual-coded row planted at MAIS08=3


def test_headline_safe_mask_reads_v2_event_floor():
    """The v2 power floor is enforced by the same headline_safe machinery the
    v1 rows use: a severity row below SEV_EVENT_FLOOR unweighted events (or
    without a single design df) is never headline-safe, and the v1
    discordant-pair path is untouched."""
    import json

    from crashgap.analysis import headline_safe_mask
    from crashgap.severity_codebook import SEV_EVENT_FLOOR

    rows = pd.DataFrame({"cohort_def": [
        json.dumps({"n_events": SEV_EVENT_FLOOR, "df": 20}),
        json.dumps({"n_events": SEV_EVENT_FLOOR - 1, "df": 20}),
        json.dumps({"n_events": 500, "df": 0}),
        json.dumps({"n_discordant_pairs": 29}),
        json.dumps({"n_discordant_pairs": 30}),
        json.dumps({}),  # v0 row: no split serialized, always safe
    ]})
    assert list(headline_safe_mask(rows)) == [True, False, False, False, True, True]


def test_svylogit_or_ci_contract():
    fit = SvyLogit(feature_cols=("female",), coef={"female": 0.4},
                   se={"female": 0.1}, df=20, n_obs=100, n_cases=90,
                   n_psus=24, n_strata=4, n_events=40, sum_weight=1e4,
                   kish_neff=70.0, weight_max_over_median=5.0, converged=True)
    lo, hi = fit.or_ci("female")
    t20 = 2.0859634472658364
    assert lo == pytest.approx(np.exp(0.4 - t20 * 0.1), abs=1e-3)
    assert hi == pytest.approx(np.exp(0.4 + t20 * 0.1), abs=1e-3)
    assert fit.or_ci("missing") == (None, None)
