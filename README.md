# CrashGap

Do women die more often than men in the same car crash? CrashGap recomputes the answer from
public NHTSA crash data and shows it at [crashgap.jakubwaller.eu](https://crashgap.jakubwaller.eu).
When NHTSA publishes a new year, the numbers update.

The idea came from reading [*Invisible Women*](https://en.wikipedia.org/wiki/Invisible_Women:_Exposing_Data_Bias_in_a_World_Designed_for_Men) by Caroline Criado Perez.

## The short answer

Take US fatal frontal crashes where a man and a woman sat belted in the front seats of the same
car, and count who died. Pooled over FARS 2015–2024 (22,721 such pairs), the woman was the one
who died 1.09 times as often as the man.

Most of that comes from the seat. On average the right-front passenger seat is the more
dangerous seat, and the woman was the passenger in 73% of these pairs. Split by who drove, the
ratio flips:

| who sat where | pairs | she died : he died |
|---|---:|---:|
| she is the passenger | 16,656 | 1.15 |
| she drives | 6,065 | 0.94 |

Taking the two apart (Evans' double-pair method):

| effect | frontal core | 95% CI |
|---|---:|---|
| being female, same seat | +4.1% | −1.5% to +10.1% |
| sitting in the passenger seat | +10.8% | +4.7% to +17.1% |

On this window the seat effect is clear and the female effect could be zero. In strict head-on
crashes the female effect is +11.5% and clear, and there the driver's seat is the worse one. So
the female effect in a fatal frontal crash sits somewhere between low single digits and low
double digits, depending on crash geometry. Quoting only one of those rows would be misleading.

The crash-test-dummy debate quotes a bigger number, about 1.5 times the odds of a serious injury
(Bose 2011). That is a different question on different data. CrashGap measures it too, see the
injury section below.

## What else the site shows

### Does the gap shrink with newer cars?

Partly. On FARS 2000–2024, adjusted for age, cars built before 2000 carry a female penalty of
1.26 [1.19–1.33] and 2000s cars 1.09 [1.03–1.15]. That drop holds under every specification.
After the 2000s nothing declines further. The published decline into the newest cars (down to
5.8%) is not found in this data. Whether that is a real plateau or the 2020–2024 years this
window adds cannot be told apart in this design.

### Is the passenger seat worse for women specifically?

A man and a woman in one car cannot answer this, because in every such pair she is the woman. So
cars with two men are compared with cars with two women. The driver dies more often among two men, and the passenger among two women. Most of
that raw difference is age, because in a two-women car the passenger is far more often much older
than the driver. Adjusted for age, a female-specific passenger penalty of roughly +10% to +23%
remains. This comparison rests on a weaker design than the rest of the site, and the site says so
next to the number.

### The injury gap

FARS only knows who died. CISS (2017–2024) and NASS-CDS (2000–2015) are crash-investigation
samples with graded injuries and reconstructed crash forces, which is where the published numbers
come from. Fitted with the same kind of model those studies use and adjusted for crash force
(delta-V), the female odds of injury land close to them: 1.50 [1.10–2.04] for a serious injury
(MAIS 3+, NASS-CDS) against Bose's 1.47, and 1.91 [1.23–2.96] for a moderate one (MAIS 2+, CISS)
against Craig's ~1.75. That is not a replication, since cohorts, frontal definitions and
injury-coding revisions differ from those studies, and the two sources are never pooled with each
other or with FARS. Adjusting for height and BMI does not remove the gap. Two things weaken it
and are measured on the site: single cases carry very large survey weights, and the injury
coding revision alone moves the MAIS 3+ number.

## How it is computed

Only cars with exactly one eligible man and one eligible woman enter (both belted, front outboard,
age 16–96, light vehicle), so the crash, the car and the impact are the same for both. FARS
records fatal crashes only, so every number compares who died given that someone did. None of
them is a population rate. Frontal means IMPACT1 in {11, 12, 1}; a wider fan and the head-on cut
are sensitivity variants. The seat-split intervals resample whole crashes. The age-adjusted
numbers come from a within-car conditional logit with crash-clustered standard errors, the injury
numbers from a design-weighted logistic regression with survey-design variance. The full design
and every sensitivity check are in [docs/v1-design.md](docs/v1-design.md) and
[docs/v2-design.md](docs/v2-design.md).

The finding itself is settled literature (Evans 1988; Bose 2011; Forman 2019; Atwood, Noh & Craig
2023). This repo adds the live, versioned form: every estimate lands in a SQLite `results` table
with its run timestamp, git commit, data years and serialized cohort definition.

## The results table

| estimand prefix | what it holds |
|---|---|
| `fars_doublepair_*` | the seat-split estimates; the pooled row is named `_seatconfounded` so nobody quotes it as the headline |
| `fars_condlogit_*` | age-adjusted female effect per model-year band, the trend slope, and the shared seat and age terms; `_separatenuisance`, `_agepiecewise` and `_vehage12` are sensitivity refits; `_pooled` collapses the bands and is not a substitute for the seat-balanced +4.1% |
| `fars_samesex_*` | passenger-vs-driver odds inside two-men and two-women cars and their log-ratio; quote the `_ageadj` rows together with `_agecomparable` |
| `{ciss,nass}_svylogit_*` | female injury odds ratios for MAIS 2+ and 3+ in the `base`, `dv` and `dvanthro` tiers, plus `_wtrim95` and the AIS coding pair |

A row backed by fewer than 30 discordant pairs (30 outcome events for the injury rows) is written
but never headlined. Check `n_discordant_pairs` in `cohort_def` before quoting a number straight
from SQL; `n_pairs` counts the whole eligible cohort and overstates power.

## Usage

```bash
pip install -e .                       # pandas, statsmodels, streamlit, altair
crashgap ingest --years 2000-2024      # ~470 MB of zips from static.nhtsa.gov, cached in data/raw
crashgap analyze --years 2015-2024     # modern window, the dashboard headline
crashgap analyze --years 2000-2024     # full window, the trend sections
crashgap analyze-v1 --years 2015-2024  # age-adjusted rows, modern window
crashgap analyze-v1 --years 2000-2024  # age-adjusted rows, full window
crashgap ingest-severity               # CISS + NASS-CDS
crashgap analyze-v2                    # injury odds ratios
crashgap dashboard                     # Streamlit on :8501
```

Everything lands in `data/crashgap.db` and every step is idempotent. The dashboard reads the
latest run per estimand family and window, so both windows need both `analyze` and `analyze-v1`.

## Refresh

FARS and CISS each publish one new year a year. `deploy/refresh.sh` re-pulls the newest years and
re-runs every analysis. Point a monthly cron at it.

## Data

NHTSA FARS National CSV releases, 2000–2024, US public domain. Two FARS recodes sit inside that
span: the unknown-age code changed in 2009 and the impact-point codes in 2010. Both were checked
against the real files and are harmless for the core cohort. The wide frontal variant selects a
somewhat narrower crash set after 2010, so cross-era comparisons of that one variant carry an
asterisk (see `crashgap/codebook.py`). The newest FARS year is an annual report file and gets
revised before final release. CISS and NASS-CDS are US public domain as well.

## License

MIT. The data is US-federal public domain.
