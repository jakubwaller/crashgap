import zipfile

from crashgap.ingest import find_member, load_year


def make_zip(path, nested: bool, case_mix: bool):
    """A minimal FARS-shaped zip; column case and member layout deliberately vary."""
    accident = "ST_CASE,STATE,MAN_COLL,HARM_EV,VE_TOTAL,PERSONS,FATALS\n10001,1,2,12,1,2,1\n"
    vehicle = ("st_case,veh_no,body_typ,mod_year,impact1,deformed,numoccs\n"
               "10001,1,4,2015,12,6,2\n")   # ROLLOVER column intentionally absent
    person = ("ST_CASE,VEH_NO,PER_NO,PER_TYP,SEX,AGE,SEAT_POS,REST_USE,AIR_BAG,INJ_SEV\n"
              "10001,1,1,1,1,44,11,3,1,0\n"
              "10001,1,2,2,2,41,13,3,1,4\n")
    prefix = "FARS9999NationalCSV/" if nested else ""
    names = ["accident.CSV", "Vehicle.csv", "person.csv"] if case_mix else \
            ["accident.csv", "vehicle.csv", "person.csv"]
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in zip(names, (accident, vehicle, person)):
            zf.writestr(prefix + name, content)


def test_find_member_is_case_and_path_insensitive(tmp_path):
    zp = tmp_path / "fars.zip"
    make_zip(zp, nested=True, case_mix=True)
    with zipfile.ZipFile(zp) as zf:
        assert find_member(zf, "accident").endswith("accident.CSV")
        assert find_member(zf, "vehicle").endswith("Vehicle.csv")


def test_load_year_and_missing_columns(conn, tmp_path):
    zp = tmp_path / "fars.zip"
    make_zip(zp, nested=False, case_mix=False)
    counts = load_year(conn, 2099, zp)
    assert counts == {"accident": 1, "vehicle": 1, "person": 2}
    # absent ROLLOVER column lands as NULL, not a crash
    rollover = conn.execute(
        "SELECT rollover FROM vehicle WHERE year=2099").fetchone()[0]
    assert rollover is None
    # re-ingest is idempotent on the natural key
    load_year(conn, 2099, zp)
    n = conn.execute("SELECT COUNT(*) FROM person WHERE year=2099").fetchone()[0]
    assert n == 2
    # the loaded pair flows through the whole pipeline
    row = conn.execute(
        "SELECT is_female, died FROM v_occupant WHERE year=2099 AND per_no=2").fetchone()
    assert row == (1, 1)
