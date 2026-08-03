"""Command line interface.

  crashgap ingest --years 2015-2024     download + load FARS into SQLite
  crashgap analyze --years 2015-2024    double-pair estimates -> results table
  crashgap dashboard                    start the Streamlit dashboard
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_DB = "data/crashgap.db"
DEFAULT_RAW = "data/raw"


def parse_years(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="crashgap",
        description="Female-vs-male road-crash fatality gap from NHTSA FARS.",
    )
    parser.add_argument("--db", default=DEFAULT_DB, help=f"SQLite path (default {DEFAULT_DB})")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="download + load FARS years")
    p_ingest.add_argument("--years", default="2015-2024", help="e.g. 2015-2024 or 2022")
    p_ingest.add_argument("--data-dir", default=DEFAULT_RAW,
                          help=f"where zips are cached (default {DEFAULT_RAW})")

    p_analyze = sub.add_parser("analyze", help="run the double-pair estimates")
    p_analyze.add_argument("--years", default="2015-2024")
    p_analyze.add_argument("--reps", type=int, default=2000,
                           help="bootstrap replicates (default 2000)")

    sub.add_parser("dashboard", help="start the Streamlit dashboard")

    args = parser.parse_args(argv)

    from .db import connect

    if args.command == "ingest":
        from .ingest import ingest_years
        years = parse_years(args.years)
        conn = connect(args.db)
        ingest_years(conn, years, args.data_dir)
    elif args.command == "analyze":
        from .analysis import analyze, by_model_year_band
        years = parse_years(args.years)
        span = (years[0], years[-1])
        conn = connect(args.db)

        def pct(x: float | None) -> str:
            return f"{(x - 1) * 100:+.1f}%" if x else "n/a"

        for dp, dec in analyze(conn, span, reps=args.reps):
            print(f"\n--- frontal={dp.frontal} "
                  f"({dp.n_pairs:,} pairs, {dp.n_crashes:,} crashes) ---")
            print(f"  pooled (SEAT-CONFOUNDED, do not report alone): RR={dp.rr} "
                  f"[{dp.ci_lo}, {dp.ci_hi}]")
            print(f"    she is passenger: RR={dec.rr_she_passenger} "
                  f"({dec.n_pairs_she_passenger:,} pairs)")
            print(f"    she drives:       RR={dec.rr_she_drives} "
                  f"({dec.n_pairs_she_drives:,} pairs)")
            sig = "significant" if dec.sex_is_significant else "NOT significant"
            print(f"  sex effect  (seat-balanced): {dec.sex_effect} = {pct(dec.sex_effect)} "
                  f"{dec.sex_ci} -> {sig}")
            sig = "significant" if dec.seat_is_significant else "NOT significant"
            print(f"  seat effect (right front):   {dec.seat_effect} = {pct(dec.seat_effect)} "
                  f"{dec.seat_ci} -> {sig}")
        print("\nmodel-year bands (frontal=core, pooled ratio, no trend claim):")
        print(by_model_year_band(conn, span).to_string(index=False))
    elif args.command == "dashboard":
        app = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
        os.environ["CRASHGAP_DB"] = args.db
        os.execvp(sys.executable, [sys.executable, "-m", "streamlit", "run", str(app)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
