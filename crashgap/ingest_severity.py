"""Download and load CISS (2017+) and NASS-CDS (2000-2015) into sev_occupant.

CISS ships one CSV zip per year; NASS-CDS ships loose per-table sas7bdat
files whose folder migrated over the years (Formatted Data/ for 2009+,
PCSAS/ for 2001-2008, and 2000 only inside PCSAS/PCSAS.zip), so the
downloader tries each layout in order and caches whatever it finds as
data/raw/nass/{year}_{table}.sas7bdat. The raw files stay the source of
truth; normalization (sentinel NULLing, code collapsing, the frontal
classifier) happens here, once, in plain pandas on the way into SQLite.
"""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

import pandas as pd
import requests

from . import severity_codebook as scb
from .db import insert_frame

CISS_URL = "https://static.nhtsa.gov/nhtsa/downloads/CISS/{year}/CISS_{year}_CSV_files.zip"
NASS_TABLE_URLS = (
    "https://static.nhtsa.gov/nhtsa/downloads/NASS/{year}/Formatted Data/{table}.sas7bdat",
    "https://static.nhtsa.gov/nhtsa/downloads/NASS/{year}/PCSAS/{table}.sas7bdat",
)
NASS_BUNDLE_URLS = (
    "https://static.nhtsa.gov/nhtsa/downloads/NASS/{year}/PCSAS/PCSAS.zip",
    "https://static.nhtsa.gov/nhtsa/downloads/NASS/{year}/PCSAS/PC{year}.zip",
)
NASS_TABLES = ("accident", "oa", "gv", "ve")

SEV_COLUMNS = [
    "source", "year", "psu", "psustrat", "case_id", "veh_no", "occ_no",
    "weight", "sex_female", "age", "seat_pos", "belted",
    "height", "body_weight", "bmi", "mais", "mais08",
    "body_typ", "mod_year", "dv_total", "is_frontal",
]


def _download(url: str, dest: Path) -> bool:
    """Fetch url to dest unless cached; False on 404 (caller tries the next
    layout), raise on anything else."""
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as resp:
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        tmp.rename(dest)
    return True


# --------------------------------------------------------------------------
# Normalizers - pure frame -> column transforms, unit-testable without files.
# --------------------------------------------------------------------------


def sex_female(sex: pd.Series) -> pd.Series:
    """1 for any female code (incl. the four pregnancy codes), 0 for male,
    NA otherwise - both sources share this coding."""
    out = pd.Series(pd.NA, index=sex.index, dtype="object")
    out[sex.isin(scb.SEV_SEX_FEMALE)] = 1
    out[sex == scb.SEV_SEX_MALE] = 0
    return out


def belted_flag(belt: pd.Series) -> pd.Series:
    """1 for {shoulder, lap, lap+shoulder}, 0 for an affirmative non-use (0),
    NA for everything else (inoperative, type unknown, child-seat combos,
    unknown, missing)."""
    out = pd.Series(pd.NA, index=belt.index, dtype="object")
    out[belt.isin(scb.SEV_BELTED)] = 1
    out[belt == 0] = 0
    return out


def valid_mais(mais: pd.Series) -> pd.Series:
    """MAIS 0-6 kept, every unknown code (7 in AIS90, 9/99 elsewhere, SAS
    missing) -> NA."""
    return mais.where(mais.isin(scb.SEV_MAIS_VALID), pd.NA)


def _null_sentinel(s: pd.Series, sentinel: float) -> pd.Series:
    return s.where(s != sentinel, pd.NA)


def _bounded(s: pd.Series, lo: float, hi: float) -> pd.Series:
    """NA outside [lo, hi] - catches both sources' unknown sentinels and the
    occasional impossible entry with one rule."""
    return s.where((s >= lo) & (s <= hi), pd.NA)


def ciss_frontal(cdc: pd.DataFrame) -> pd.DataFrame:
    """(CASENUMBER, VEHNO) -> is_frontal from the vehicle's primary damage
    event: the lowest DVRANK in 1-7, ties and rank-unknowns broken by lowest
    EVENTNO. Plane F with clock 11/12/1."""
    ranked = cdc.assign(
        _rank=cdc["DVRANK"].where(cdc["DVRANK"].between(1, 7), 99),
    ).sort_values(["CASENUMBER", "VEHNO", "_rank", "EVENTNO"])
    # drop_duplicates keeps the first ROW per vehicle; groupby(...).first()
    # would take the first non-NaN per COLUMN and could chimera fields from
    # two different events if a primary row ever carried a NaN.
    primary = ranked.drop_duplicates(subset=["CASENUMBER", "VEHNO"], keep="first").copy()
    primary["is_frontal"] = (
        (primary["CDCPLANE"] == scb.SEV_FRONTAL_PLANE)
        & primary["OCLOCK"].isin(scb.SEV_FRONTAL_CLOCK)
    ).astype(int)
    return primary[["CASENUMBER", "VEHNO", "is_frontal"]]


def nass_frontal(ve: pd.DataFrame) -> pd.DataFrame:
    """(PSU, CASENO, VEHNO) -> is_frontal from ve's principal impact:
    GAD1 = 'F' with DOF1 in 11/12/1."""
    # explicit bytes decode: on pandas 2.x, .astype(str) stringifies b'F' to
    # "b'F'", which would silently classify every vehicle non-frontal (the
    # production read_sas path returns str, but never rely on that)
    gad = ve["GAD1"].map(
        lambda v: v.decode("latin-1") if isinstance(v, (bytes, bytearray)) else v)
    gad = gad.astype(str).str.strip()
    out = ve[["PSU", "CASENO", "VEHNO"]].copy()
    out["is_frontal"] = (
        (gad == scb.SEV_FRONTAL_PLANE) & ve["DOF1"].isin(scb.SEV_FRONTAL_CLOCK)
    ).astype(int)
    # one row per vehicle already, but be safe against duplicates
    return out.groupby(["PSU", "CASENO", "VEHNO"], as_index=False)["is_frontal"].max()


def _finish(frame: pd.DataFrame, source: str, year: int) -> pd.DataFrame:
    frame = frame.copy()
    frame.insert(0, "source", source)
    frame.insert(1, "year", year)
    frame = frame[SEV_COLUMNS]
    return frame.astype(object).where(pd.notna(frame), None)


# --------------------------------------------------------------------------
# CISS
# --------------------------------------------------------------------------


def ciss_zip_path(data_dir: Path, year: int) -> Path:
    return Path(data_dir) / "ciss" / f"CISS_{year}_CSV_files.zip"


def _member(zf: zipfile.ZipFile, base: str) -> str:
    """Member name for {base}.csv regardless of case - the zips mix OCC.CSV,
    CRASH.csv and OCC.csv across (and even within) years."""
    for name in zf.namelist():
        if name.replace("\\", "/").rsplit("/", 1)[-1].lower() == f"{base.lower()}.csv":
            return name
    raise FileNotFoundError(f"{base}.csv not found in {zf.filename}")


def load_ciss_year(conn: sqlite3.Connection, year: int, zip_file: Path | str) -> int:
    with zipfile.ZipFile(zip_file) as zf:
        occ = pd.read_csv(zf.open(_member(zf, "OCC")), encoding="latin-1", low_memory=False)
        gv = pd.read_csv(zf.open(_member(zf, "GV")), encoding="latin-1", low_memory=False)
        cdc = pd.read_csv(zf.open(_member(zf, "CDC")), encoding="latin-1", low_memory=False)

    veh = gv[["CASENUMBER", "VEHNO", "BODYTYPE", "MODELYR", "DVTOTAL"]].merge(
        ciss_frontal(cdc), on=["CASENUMBER", "VEHNO"], how="left")
    veh["is_frontal"] = veh["is_frontal"].fillna(0).astype(int)

    df = occ.merge(veh, on=["CASENUMBER", "VEHNO"], how="inner")
    height = _bounded(_null_sentinel(df["HEIGHT"], scb.CISS_HEIGHT_UNKNOWN),
                      *scb.SEV_HEIGHT_RANGE)
    body_weight = _bounded(_null_sentinel(df["WEIGHT"], scb.CISS_WEIGHT_UNKNOWN),
                           *scb.SEV_WEIGHT_RANGE)
    out = pd.DataFrame({
        "psu": df["PSU"], "psustrat": df["PSUSTRAT"],
        "case_id": df["CASENUMBER"].astype(str),
        "veh_no": df["VEHNO"], "occ_no": df["OCCNO"],
        "weight": df["CASEWGT"],
        "sex_female": sex_female(df["SEX"]),
        "age": _null_sentinel(df["AGE"], scb.CISS_AGE_UNKNOWN),
        "seat_pos": df["SEATLOC"],
        "belted": belted_flag(df["BELTUSE"]),
        "height": height, "body_weight": body_weight,
        "bmi": _null_sentinel(df["BMI"], scb.CISS_BMI_UNKNOWN),
        "mais": valid_mais(df["MAIS"]),
        "mais08": pd.NA,  # CISS codes AIS2015 in MAIS; no dual coding
        "body_typ": df["BODYTYPE"],
        "mod_year": _null_sentinel(df["MODELYR"], scb.SEV_MOD_YEAR_UNKNOWN),
        "dv_total": _null_sentinel(df["DVTOTAL"], scb.CISS_DV_UNKNOWN),
        "is_frontal": df["is_frontal"],
    })
    frame = _finish(out, "ciss", year)
    return insert_frame(conn, "sev_occupant", SEV_COLUMNS,
                        list(frame.itertuples(index=False, name=None)))


def ingest_ciss(conn: sqlite3.Connection, years: list[int],
                data_dir: Path | str) -> None:
    for year in years:
        dest = ciss_zip_path(Path(data_dir), year)
        if not _download(CISS_URL.format(year=year), dest):
            print(f"ciss {year}: not published (404), skipped")
            continue
        n = load_ciss_year(conn, year, dest)
        print(f"ciss {year}: occupants={n:,}")


# --------------------------------------------------------------------------
# NASS-CDS
# --------------------------------------------------------------------------


def nass_table_path(data_dir: Path, year: int, table: str) -> Path:
    return Path(data_dir) / "nass" / f"{year}_{table}.sas7bdat"


def fetch_nass_table(year: int, table: str, data_dir: Path) -> Path:
    dest = nass_table_path(data_dir, year, table)
    if dest.exists():
        return dest
    for pattern in NASS_TABLE_URLS:
        if _download(pattern.format(year=year, table=table), dest):
            return dest
    # bundle fallback (2000 ships only PCSAS.zip)
    for pattern in NASS_BUNDLE_URLS:
        bundle = Path(data_dir) / "nass" / f"bundle_{year}.zip"
        if _download(pattern.format(year=year), bundle):
            with zipfile.ZipFile(bundle) as zf:
                member = next((m for m in zf.namelist()
                               if m.lower().rsplit("/", 1)[-1] == f"{table}.sas7bdat"), None)
                if member is None:
                    continue
                dest.write_bytes(zf.read(member))
            return dest
    raise FileNotFoundError(f"NASS {year} {table}.sas7bdat not found in any known layout")


def load_nass_year(conn: sqlite3.Connection, year: int, data_dir: Path | str) -> int:
    data_dir = Path(data_dir)

    def read(table: str) -> pd.DataFrame:
        return pd.read_sas(fetch_nass_table(year, table, data_dir),
                           format="sas7bdat", encoding="latin-1")

    acc, oa, gv, ve = (read(t) for t in NASS_TABLES)

    veh = gv[["PSU", "CASENO", "VEHNO", "BODYTYPE", "MODELYR", "DVTOTAL"]].merge(
        nass_frontal(ve), on=["PSU", "CASENO", "VEHNO"], how="left")
    veh["is_frontal"] = veh["is_frontal"].fillna(0).astype(int)

    df = oa.merge(veh, on=["PSU", "CASENO", "VEHNO"], how="inner").merge(
        acc[["PSU", "CASENO", "PSUSTRAT"]].drop_duplicates(),
        on=["PSU", "CASENO"], how="left")

    height = _bounded(df["HEIGHT"], *scb.SEV_HEIGHT_RANGE)
    body_weight = _bounded(df["WEIGHT"], *scb.SEV_WEIGHT_RANGE)
    bmi = body_weight / (height / 100.0) ** 2
    mais08 = valid_mais(df["MAIS08"]) if "MAIS08" in df.columns else pd.NA
    out = pd.DataFrame({
        "psu": df["PSU"], "psustrat": df["PSUSTRAT"],
        "case_id": df["CASENO"].astype("Int64").astype(str),
        "veh_no": df["VEHNO"], "occ_no": df["OCCNO"],
        "weight": df["RATWGT"],
        "sex_female": sex_female(df["SEX"]),
        "age": df["AGE"],
        "seat_pos": df["SEATPOS"],
        "belted": belted_flag(df["MANUSE"]),
        "height": height, "body_weight": body_weight, "bmi": bmi,
        "mais": valid_mais(df["MAIS"]),
        "mais08": mais08,
        "body_typ": df["BODYTYPE"],
        "mod_year": _null_sentinel(df["MODELYR"], scb.SEV_MOD_YEAR_UNKNOWN),
        "dv_total": df["DVTOTAL"],
        "is_frontal": df["is_frontal"],
    })
    frame = _finish(out, "nass", year)
    return insert_frame(conn, "sev_occupant", SEV_COLUMNS,
                        list(frame.itertuples(index=False, name=None)))


def ingest_nass(conn: sqlite3.Connection, years: list[int],
                data_dir: Path | str) -> None:
    for year in years:
        n = load_nass_year(conn, year, data_dir)
        print(f"nass {year}: occupants={n:,}")
