from crashgap import codebook as cb
from crashgap.db import connect


def test_schema_is_idempotent(tmp_path):
    path = tmp_path / "x.db"
    c1 = connect(path)
    c1.close()
    c2 = connect(path)   # second connect re-runs executescript on same file
    tables = {r[0] for r in c2.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    assert {"accident", "vehicle", "person", "results", "v_occupant"} <= tables


def test_codebook_sets_are_disjoint_and_light():
    assert not (cb.BODY_TYP_PASSENGER_CAR & cb.BODY_TYP_LIGHT_TRUCK_VAN)
    assert max(cb.BODY_TYP_LIGHT_VEHICLE) < 50   # buses/heavy trucks start at 50
    assert cb.IMPACT1_FRONTAL_CORE < cb.IMPACT1_FRONTAL_WIDE
    assert cb.REST_USE_BELTED == {1, 2, 3}


def test_cohort_def_serializes(tmp_path):
    import json
    for variant in ("core", "wide", "headon"):
        parsed = json.loads(cb.cohort_def(variant))
        assert "double-pair" in parsed["design"]
