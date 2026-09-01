"""Pinned FARS code sets.

Every inclusion rule the analysis uses is a literal here, so the cohort
definition can be serialized into results.cohort_def and audited against the
FARS Analytical User's Manual. Definitions that copy a published study cite it.

Codings verified against the real 2015/2019/2022/2024 National CSV files,
and re-verified across the 2000-2014 ingestion (drift check 2026-09-01 over
the 2000/2004/2008/2009/2010/2014/2015 files). Where a FARS recode happened
inside 2000-2024, the note sits next to the affected code set below.
"""

from __future__ import annotations

import json

# SEX: 1=Male, 2=Female; 8 (2010+) and 9 = not reported/unknown -> excluded.
SEX_MALE = 1
SEX_FEMALE = 2

# PER_TYP: 1 = driver, 2 = passenger of an in-transport vehicle.
PER_TYP_OCCUPANTS = {1, 2}

# SEAT_POS: 11 = front left (driver), 13 = front right. Front outboard only,
# so both pair members face comparable restraint/airbag environments. Seat is
# the one thing that structurally differs WITHIN every pair and it is heavily
# collinear with sex (she is the passenger in ~73% of mixed pairs), so the
# analysis decomposes it out via Evans' geometric mean over the two seat
# configurations rather than pooling over it.
SEAT_POS_DRIVER = 11
SEAT_POS_RIGHT_FRONT = 13
SEAT_POS_FRONT_OUTBOARD = {SEAT_POS_DRIVER, SEAT_POS_RIGHT_FRONT}

# REST_USE belted = {1, 2, 3}: shoulder only, lap only, lap + shoulder.
# The full belted set - not 3 alone, which silently drops lap-only and
# shoulder-only occupants. Cross-era stable: the 2010 REST_USE recode moved
# "none used" from 0 to 7 and reshuffled the helmet codes, but 1/2/3 keep
# their meaning throughout 2000-2024; "restraint used, type unknown" (8) is
# excluded in both eras.
REST_USE_BELTED = {1, 2, 3}

# INJ_SEV (KABCO): 4 = fatal. 0-3 = survived spectrum; 5/6/9 (injured
# severity unknown / died prior / unknown) are excluded from the outcome.
INJ_SEV_FATAL = 4
INJ_SEV_NONFATAL = {0, 1, 2, 3}

# Age window per Atwood, Noh & Craig 2023 (FARS 1975-2019 double-pair study).
# The unknown-age sentinel changed schemes in 2009 (99 through 2008, 998/999
# from 2009 on); both fall outside 16-96 and drop out with the window, which
# is presumably why the published study pinned 96 - do not widen the top end
# without handling the pre-2009 99 sentinel explicitly.
AGE_MIN = 16
AGE_MAX = 96

# IMPACT1 clock points. Primary frontal definition pinned to the 11-1 o'clock
# core zone used by Forman et al. 2019; the wider fan and the strict
# MAN_COLL=2 head-on cut are documented sensitivity variants, never silently
# substituted.
#
# Cross-era caveat for the WIDE variant only: 2010+ files add side-specific
# codes (61-63 left, 81-83 right) that drain crashes out of clock points
# 2/3/9/10, so the wide fan's 10 and 2 select a somewhat narrower set of
# crashes after 2010 than before. The core zone {11, 12, 1} shows no such
# break (volumes continuous across 2009->2010), and MAN_COLL=2 keeps its
# meaning through the 2010 recode ("Head-On" -> "Front-to-Front", same code,
# continuous volumes) - only cross-era comparisons of the wide variant carry
# this asterisk.
IMPACT1_FRONTAL_CORE = {11, 12, 1}
IMPACT1_FRONTAL_WIDE = {10, 11, 12, 1, 2}
MAN_COLL_HEAD_ON = 2

# BODY_TYP light-vehicle grouping (passenger cars + light trucks/vans,
# GVWR <= 10,000 lbs), matching the NHTSA Traffic Safety Facts grouping.
# Names from the 2022 vehicle.csv BODY_TYPNAME column. Codes 30-33 are the
# pre-2020 pickup split; recent years collapse them into 34 (Light Pickup).
BODY_TYP_PASSENGER_CAR = {
    1,   # Convertible
    2,   # 2-door sedan, hardtop, coupe
    3,   # 3-door/2-door hatchback
    4,   # 4-door sedan, hardtop
    5,   # 5-door/4-door hatchback
    6,   # Station wagon
    7,   # Hatchback, number of doors unknown
    8,   # Sedan/hardtop, number of doors unknown
    9,   # Other or unknown automobile type
    10,  # Auto-based pickup
    11,  # Auto-based panel
    12,  # Large limousine
    13,  # Three-wheel automobile or automobile derivative
    17,  # 3-door coupe
}
BODY_TYP_LIGHT_TRUCK_VAN = {
    14,  # Compact utility
    15,  # Large utility
    16,  # Utility station wagon
    19,  # Utility vehicle, unknown body type
    20,  # Minivan
    21,  # Large van
    22,  # Step van or walk-in van
    28,  # Other van type
    29,  # Unknown van type
    30,  # Compact pickup (pre-2020 coding)
    31,  # Standard pickup (pre-2020 coding)
    32,  # Pickup with slide-in camper (pre-2020 coding)
    33,  # Convertible pickup (pre-2020 coding)
    34,  # Light pickup
    39,  # Unknown (pickup style) light conventional truck
    40,  # Cab chassis based
    41,  # Truck-based panel
    45,  # Other light conventional truck
    48,  # Unknown light truck type
    49,  # Unknown light vehicle type
}
BODY_TYP_LIGHT_VEHICLE = BODY_TYP_PASSENGER_CAR | BODY_TYP_LIGHT_TRUCK_VAN

FRONTAL_VARIANTS = {
    "core": "IMPACT1 in {11,12,1} (Forman 2019 frontal core)",
    "wide": "IMPACT1 in {10,11,12,1,2} (wide frontal fan)",
    "headon": "MAN_COLL = 2 (head-on collision manner)",
}


def sql_in(values: set[int]) -> str:
    """Render a code set as a SQL IN-list, sorted for stable view definitions."""
    return ", ".join(str(v) for v in sorted(values))


def cohort_def(frontal: str) -> str:
    """The serialized inclusion rules stored next to every result row."""
    return json.dumps(
        {
            "occupants": "PER_TYP in [1,2]",
            "seat": "SEAT_POS in [11,13] (front outboard)",
            "belted": "REST_USE in [1,2,3]",
            "frontal": FRONTAL_VARIANTS[frontal],
            "light_vehicle": f"BODY_TYP in [{sql_in(BODY_TYP_LIGHT_VEHICLE)}]",
            "age": f"{AGE_MIN}-{AGE_MAX} (Atwood/Noh/Craig 2023 window)",
            "outcome": "INJ_SEV=4 fatal vs 0-3 survived; 5/6/9 excluded",
            "design": "Evans double-pair: vehicles with exactly one eligible male "
                      "and one eligible female; RR from discordant pairs",
        },
        sort_keys=True,
    )
