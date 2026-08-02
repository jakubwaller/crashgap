"""Download and load FARS National CSV releases into SQLite.

One zip per year from
https://static.nhtsa.gov/nhtsa/downloads/FARS/{YEAR}/National/FARS{YEAR}NationalCSV.zip

The zips are not uniform across years: members sit in the root or in a
subfolder, and file names switch case (accident.CSV, Person.CSV, vehicle.csv
all occur), so member lookup is case- and path-insensitive. Columns are read
case-insensitively and missing columns land as NULL (e.g. PERSONS disappears
from accident.csv in some years).
"""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

import pandas as pd
import requests

from .db import insert_frame

URL_TEMPLATE = "https://static.nhtsa.gov/nhtsa/downloads/FARS/{year}/National/FARS{year}NationalCSV.zip"

# table -> raw FARS columns kept (upper-case as in the codebook)
COLUMNS = {
    "accident": ["ST_CASE", "STATE", "MAN_COLL", "HARM_EV", "VE_TOTAL", "PERSONS", "FATALS"],
    "vehicle": ["ST_CASE", "VEH_NO", "BODY_TYP", "MOD_YEAR", "IMPACT1",
                "DEFORMED", "NUMOCCS", "ROLLOVER"],
    "person": ["ST_CASE", "VEH_NO", "PER_NO", "PER_TYP", "SEX", "AGE",
               "SEAT_POS", "REST_USE", "AIR_BAG", "INJ_SEV"],
}


def zip_path(data_dir: Path, year: int) -> Path:
    return Path(data_dir) / f"FARS{year}NationalCSV.zip"


def download(year: int, data_dir: Path | str, force: bool = False) -> Path:
    """Fetch one year's zip unless it is already on disk."""
    dest = zip_path(Path(data_dir), year)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = URL_TEMPLATE.format(year=year)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        tmp.rename(dest)
    return dest


def find_member(zf: zipfile.ZipFile, base: str) -> str:
    """Locate {base}.csv regardless of folder nesting and case."""
    for name in zf.namelist():
        leaf = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if leaf == f"{base}.csv":
            return name
    raise FileNotFoundError(f"{base}.csv not found in {zf.filename}")


def read_table(zf: zipfile.ZipFile, base: str) -> pd.DataFrame:
    wanted = set(COLUMNS[base])
    with zf.open(find_member(zf, base)) as fh:
        df = pd.read_csv(fh, encoding="latin-1", low_memory=False,
                         usecols=lambda c: c.upper() in wanted)
    df.columns = [c.upper() for c in df.columns]
    for col in COLUMNS[base]:
        if col not in df.columns:
            df[col] = pd.NA
    return df[COLUMNS[base]]


def load_year(conn: sqlite3.Connection, year: int, zip_file: Path | str,
              source: str = "fars") -> dict[str, int]:
    """Load accident/vehicle/person for one year. Idempotent per (source, year)."""
    counts = {}
    with zipfile.ZipFile(zip_file) as zf:
        for base, cols in COLUMNS.items():
            df = read_table(zf, base)
            df.insert(0, "SOURCE", source)
            df.insert(1, "YEAR", year)
            df = df.astype(object).where(pd.notna(df), None)
            db_cols = ["source", "year"] + [c.lower() for c in cols]
            counts[base] = insert_frame(conn, base, db_cols,
                                        list(df.itertuples(index=False, name=None)))
    return counts


def ingest_years(conn: sqlite3.Connection, years: list[int],
                 data_dir: Path | str) -> None:
    for year in years:
        path = download(year, data_dir)
        counts = load_year(conn, year, path)
        print(f"{year}: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
