# CrashGap

The female-vs-male road-crash fatality gap, computed live from public-domain NHTSA
[FARS](https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars) data.

Cars are crash-tested around the median male body. The literature has measured the consequence
for decades; the number just isn't anywhere you can link to, query, or watch move as new model
years arrive. CrashGap is that missing layer: an annual pipeline, a SQLite `results` table where
every estimate carries its run timestamp, git commit, FARS years and full serialized cohort
definition, and one honest dashboard tile.

**Current estimate (FARS 2015–2024):** in belted front-outboard mixed-sex pairs riding in the
same vehicle in a fatal frontal crash, the woman died and the man survived **1.09×** as often as
the reverse (95% CI 1.04–1.15, crash-clustered bootstrap; 22,721 pairs in 21,943 crashes).

## Method, in one paragraph

FARS is a census of *fatal* crashes, so a raw pooled odds ratio is dominated by
crash-involvement selection and can point the wrong way. CrashGap therefore uses the Evans
double-pair design: only vehicles carrying exactly one eligible male and one eligible female
(both belted, front outboard, age 16–96, light vehicle) enter, which holds crash severity,
delta-V, vehicle and impact direction physically constant within the vehicle. The estimate is
the ratio of discordant outcomes — (she died, he survived) : (he died, she survived). Confidence
intervals resample whole crashes, never persons. Frontal is pinned to IMPACT1 ∈ {11, 12, 1}
(Forman 2019), with the wider fan and the head-on cut reported as sensitivity variants. Every
code set is a literal in `crashgap/codebook.py` with its source next to it.

The finding itself is settled science — Evans (1988), Bose et al. (2011), Forman et al. (2019),
Atwood, Noh & Craig (2023). What this repo adds is the live, versioned, machine-readable form,
and it says so.

What this number is **not**: a population rate (every crash here already killed someone), a
severity-adjusted odds ratio (that needs CISS — planned next rung), or evidence about the
model-year trend (this 10-year window can't resolve it; see the dashboard's caveats).

## Usage

```bash
pip install -e .
crashgap ingest --years 2015-2024    # ~300 MB of zips from static.nhtsa.gov, cached in data/raw
crashgap analyze --years 2015-2024   # double-pair estimates -> results table + stdout
crashgap dashboard                   # Streamlit tile on :8501
```

Everything lands in `data/crashgap.db`. Re-running either step is idempotent.

## Refresh

FARS publishes one new year annually (and finalizes the previous ARF release).
`deploy/refresh.sh` re-pulls the two newest years and re-runs the analysis; point a monthly cron
at it and the tile updates itself when NHTSA ships.

## Data

NHTSA FARS National CSV releases, 1975–2024, US public domain. This repo pools 2015–2024, where
the codings the analysis touches are stable. The 2024 file is an Annual Report File and gets
revised by NHTSA before final release.

## License

MIT. The data is US-federal public domain; the numbers are only as good as their caveats, which
ship on the same page.
