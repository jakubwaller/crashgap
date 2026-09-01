"""Pinned CISS / NASS-CDS code sets for the v2 severity rung.

Same contract as codebook.py: every inclusion rule the severity analysis uses
is a literal here, serialized into results.cohort_def. CISS codings verified
against the FORMAT22.sas value labels shipped in CISS_2022_SAS_files.zip;
NASS codings verified empirically across the 2000/2005/2010 files (the NASS
code families are the ancestors of the CISS ones and match on every field
this analysis touches).
"""

from __future__ import annotations

# SEX (both sources): 1 = male; female is 2 PLUS the four pregnancy codes
# (3 = pregnant 1st trimester, 4 = 2nd, 5 = 3rd, 6 = trimester unknown).
# Collapsing "female" to {2} would silently drop pregnant women - the exact
# cohort the crash-test-dummy critique cares about. 9 / missing = unknown.
SEV_SEX_MALE = 1
SEV_SEX_FEMALE = {2, 3, 4, 5, 6}

# ROLE (both): 1 = driver, 2 = passenger. Non-motorists (8) excluded.
SEV_ROLE_OCCUPANTS = {1, 2}

# Seat location (CISS SEATLOC, NASS SEATPOS): FARS convention, 11 = driver,
# 13 = right front.
SEV_SEAT_FRONT_OUTBOARD = {11, 13}

# Belt use (CISS BELTUSE, NASS MANUSE): 2 = shoulder, 3 = lap, 4 = lap and
# shoulder. 5 = "belt used - type unknown" excluded, mirroring the FARS
# convention (REST_USE 8) of excluding used-type-unknown; 12-18 are
# child-seat combinations the adult age window never reaches; 0/1 = none /
# inoperative; 99 unknown.
SEV_BELTED = {2, 3, 4}

# MAIS: 0-6 are valid severities. The injured-severity-unknown code differs
# by AIS revision: NASS MAIS (AIS90) uses 7, NASS MAIS08 and CISS MAIS use 9
# (CISS also has 99 = unknown if injured). Anything not in 0-6 -> NULL at
# ingest, so the analysis only ever sees graded occupants.
SEV_MAIS_VALID = set(range(0, 7))

# Age window: same 16-96 as the FARS channels (codebook.AGE_MIN/AGE_MAX),
# re-declared here so cohort_def for severity rows is self-contained.
SEV_AGE_MIN = 16
SEV_AGE_MAX = 96

# Frontal, symmetric with v0/v1's 11-1 o'clock core zone, on the vehicle's
# primary damage event: NASS ve GAD1 = 'F' with DOF1 in the clock set; CISS
# CDC row with the lowest DVRANK (ties/unranked by lowest EVENTNO),
# CDCPLANE = 'F' with OCLOCK in the clock set. Not identical to Bose 2011's
# frontal (GAD F with a wider PDOF fan) - the benchmark table says so.
SEV_FRONTAL_PLANE = "F"
SEV_FRONTAL_CLOCK = {11, 12, 1}

# Sentinels (CISS; NASS uses SAS missing): -> NULL at ingest.
CISS_AGE_UNKNOWN = 999
CISS_HEIGHT_UNKNOWN = 999
CISS_WEIGHT_UNKNOWN = 999
CISS_BMI_UNKNOWN = 99.9
CISS_DV_UNKNOWN = 999
SEV_MOD_YEAR_UNKNOWN = 9999

# Plausibility bounds for anthropometrics (both sources record cm / kg).
SEV_HEIGHT_RANGE = (100, 220)
SEV_WEIGHT_RANGE = (30, 250)

# Power floor for a severity fit: unweighted positive-outcome count. The v2
# analogue of analysis.POWER_FLOOR's discordant-pair floor.
SEV_EVENT_FLOOR = 30
