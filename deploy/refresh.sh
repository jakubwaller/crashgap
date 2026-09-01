#!/usr/bin/env bash
# Monthly cron: re-pull the two newest FARS years (the ARF gets revised until
# final release, and a brand-new year appears each autumn), then re-analyze
# BOTH windows x BOTH estimand families. The dashboard reads the latest run
# per (family, fars_years), so all four runs must happen here - dropping one
# would silently pin that window's numbers to an old commit.
# A 404 for a not-yet-published year is fine - the ingest keeps what it has.
set -uo pipefail
cd "$(dirname "$0")/.."

YEAR=$(date +%Y)
for y in $((YEAR - 2)) $((YEAR - 1)); do
    rm -f "data/raw/FARS${y}NationalCSV.zip"
    python -m crashgap.cli ingest --years "$y" || echo "FARS $y not published yet"
done

# Newest ingested year (the newest ARF may not be out yet); windows follow it.
LAST=$(python - <<'EOF'
import sqlite3
conn = sqlite3.connect("data/crashgap.db")
print(conn.execute("SELECT MAX(year) FROM person WHERE source='fars'").fetchone()[0])
EOF
)
FULL="2000-${LAST}"
MODERN="$((LAST - 9))-${LAST}"

for years in "$MODERN" "$FULL"; do
    python -m crashgap.cli analyze --years "$years"
    python -m crashgap.cli analyze-v1 --years "$years"
done
