#!/usr/bin/env bash
# Monthly cron: re-pull the two newest FARS years (the ARF gets revised until
# final release, and a brand-new year appears each autumn), then re-analyze.
# A 404 for a not-yet-published year is fine - the ingest keeps what it has.
set -uo pipefail
cd "$(dirname "$0")/.."

YEAR=$(date +%Y)
for y in $((YEAR - 2)) $((YEAR - 1)); do
    rm -f "data/raw/FARS${y}NationalCSV.zip"
    python -m crashgap.cli ingest --years "$y" || echo "FARS $y not published yet"
done
python -m crashgap.cli analyze --years "2015-$((YEAR - 1))" \
    || python -m crashgap.cli analyze --years "2015-$((YEAR - 2))"
