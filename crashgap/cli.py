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
        for dp in analyze(conn, span, reps=args.reps):
            ci = f"[{dp.ci_lo}, {dp.ci_hi}]" if dp.ci_lo is not None else "[-]"
            print(f"frontal={dp.frontal:6s} RR={dp.rr} {ci} "
                  f"f_only={dp.f_only} m_only={dp.m_only} "
                  f"pairs={dp.n_pairs:,} crashes={dp.n_crashes:,}")
        print("\nmodel-year bands (frontal=core):")
        print(by_model_year_band(conn, span).to_string(index=False))
    elif args.command == "dashboard":
        app = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
        os.environ["CRASHGAP_DB"] = args.db
        os.execvp(sys.executable, [sys.executable, "-m", "streamlit", "run", str(app)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
