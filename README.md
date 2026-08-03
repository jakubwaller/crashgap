# CrashGap

Female-vs-male fatality risk in the *same* crash, computed live from public-domain NHTSA
[FARS](https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars) data — with
the seat-position confound separated out instead of quietly baked in.

Cars are crash-tested around the median male body. The literature has measured the consequence for
decades; the number just isn't anywhere you can link to, query, or watch move as new model years
arrive. CrashGap is that missing layer: an annual pipeline, a SQLite `results` table where every
estimate carries its run timestamp, git commit, FARS years and full serialized cohort definition,
and a dashboard that leads with the decomposition rather than the headline-friendly number.

## The headline, and why it isn't the obvious one

Restrict to belted front-seat **mixed-sex pairs in the same vehicle** in a fatal frontal crash and
pool them, and women come out **1.09×** more likely to be the one who died (FARS 2015–2024, 22,721
pairs). That number is wrong to publish, and here is why: seat position is the one attribute that
differs within every pair, and it is ~73/27 collinear with sex — she is the passenger in 16,656 of
those pairs and the driver in only 6,065. Split by seat configuration and the ratio *flips*:

| configuration | pairs | she died : he died |
|---|---:|---:|
| she is the passenger, he drives | 16,656 | 1.153 |
| she drives, he is the passenger | 6,065 | 0.940 |

If sex alone drove the outcome those would agree. Evans' geometric means separate them — with *f*
the female multiplier and *s* the right-front multiplier, the two configurations measure *f·s* and
*f/s*, so `sqrt(product)` recovers *f* and `sqrt(quotient)` recovers *s*:

| effect | frontal core | 95% CI | significant |
|---|---:|---|---|
| being female (seat-balanced) | **+4.1%** | −1.5% to +10.1% | no |
| sitting right-front | **+10.8%** | +4.7% to +17.1% | yes |

So on FARS 2015–2024, in the primary frontal cut, **the seat matters more than the sex**, and the
1.09 pooled figure is mostly a seat effect wearing a sex label.

**The cuts disagree, and that's informative.** In the strict head-on cut (`MAN_COLL=2`) the female
effect is **+11.5% and significant**, while the seat effect *reverses* — the driver fares worse.
That's physically plausible rather than noise: head-on collisions are predominantly offset toward
the driver's side, which is why the driver-side small-overlap crash test existed for six years
before a passenger-side one was added. The honest summary is that the seat-balanced female effect
sits somewhere from low single digits to low double digits depending on crash geometry, and ten
years of FARS can't pin it down further. Quoting only the significant row is cherry-picking.

None of this refutes the larger published severity-adjusted gap (Bose 2011 reports MAIS 3+ odds
~1.47). That's a different quantity, measured on crash-investigation data with injury grading and
survey weights — the CISS rung, not this one.

## Method

Evans double-pair design: only vehicles carrying exactly one eligible male and one eligible female
(both belted, front outboard, age 16–96, light vehicle) enter, which holds crash severity, delta-V,
vehicle and impact direction physically constant within the vehicle. FARS is a census of *fatal*
crashes, so a raw occupant-level odds ratio would be dominated by crash-involvement selection and
can point the wrong way — hence the within-vehicle design, and hence the seat split on top of it.
Frontal is pinned to IMPACT1 ∈ {11, 12, 1} (Forman 2019), with the wider fan and head-on cut
reported as sensitivity variants. Every code set is a literal in `crashgap/codebook.py` with its
source beside it. Confidence intervals are percentile bootstraps resampling **whole crashes**,
never persons, with both seat strata resampled inside one replicate.

What this repo adds over the literature (Evans 1988; Bose 2011; Forman 2019; Atwood, Noh & Craig
2023) is the live, versioned, machine-readable form — not new evidence, and it says so on the page.

What these numbers are **not**: population rates (every crash here already killed someone),
severity-adjusted odds ratios, age-adjusted (women in these pairs average ~1.4 years younger), or
evidence about the model-year trend (this window can't resolve it).

## Usage

```bash
pip install -e .
crashgap ingest --years 2015-2024    # ~300 MB of zips from static.nhtsa.gov, cached in data/raw
crashgap analyze --years 2015-2024   # pooled + decomposed estimates -> results table + stdout
crashgap dashboard                   # Streamlit tile on :8501
```

Everything lands in `data/crashgap.db`. Re-running either step is idempotent. Each run writes three
estimands per frontal cut: the seat-balanced sex effect, the right-front seat effect, and the
pooled ratio explicitly named `_seatconfounded` so it can never be mistaken for the headline.

## Refresh

FARS publishes one new year annually (and finalizes the previous ARF release). `deploy/refresh.sh`
re-pulls the two newest years and re-runs the analysis; point a monthly cron at it and the numbers
update themselves when NHTSA ships.

## Roadmap

- **v1** — conditional logistic regression with occupant-age adjustment and cluster-robust SEs,
  which is what's needed to settle both the residual sex effect and the model-year trend.
- **v2** — CISS/NASS-CDS with MAIS injury grading and survey weights: the rung that reaches the
  severity-adjusted odds ratios the literature reports.

## Data

NHTSA FARS National CSV releases, 1975–2024, US public domain. This repo pools 2015–2024, where the
codings the analysis touches are stable. The 2024 file is an Annual Report File and gets revised by
NHTSA before final release.

## License

MIT. The data is US-federal public domain; the numbers are only as good as their caveats, which
ship on the same page.
