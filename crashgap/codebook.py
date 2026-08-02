"""Pinned FARS code sets.

Every inclusion rule the analysis uses is a literal here, so the cohort
definition can be serialized into results.cohort_def and audited against the
FARS Analytical User's Manual. Definitions that copy a published study cite it.

Codings verified against the real 2015/2019/2022/2024 National CSV files.
"""

from __future__ import annotations

import json

# SEX: 1=Male, 2=Female; 8 (2010+) and 9 = not reported/unknown -> excluded.
SEX_MALE = 1
SEX_FEMALE = 2

# PER_TYP: 1 = driver, 2 = passenger of an in-transport vehicle.
PER_TYP_OCCUPANTS = {1, 2}

# SEAT_POS: 11 = front left (driver), 13 = front right. Front outboard only,
# so both pair members face comparable restraint/airbag environments.
SEAT_POS_FRONT_OUTBOARD = {11, 13}

# REST_USE belted = {1, 2, 3}: shoulder only, lap only, lap + shoulder.
# The full belted set - not 3 alone, which silently drops lap-only and
# shoulder-only occupants.
REST_USE_BELTED = {1, 2, 3}

# INJ_SEV (KABCO): 4 = fatal. 0-3 = survived spectrum; 5/6/9 (injured
# severity unknown / died prior / unknown) are excluded from the outcome.
INJ_SEV_FATAL = 4
INJ_SEV_NONFATAL = {0, 1, 2, 3}

# Age window per Atwood, Noh & Craig 2023 (FARS 1975-2019 double-pair study).
# FARS sentinels 998/999 fall outside the window and drop out with it.
AGE_MIN = 16
AGE_MAX = 96

# IMPACT1 clock points. Primary frontal definition pinned to the 11-1 o'clock
# core zone used by Forman et al. 2019; the wider fan and the strict
# MAN_COLL=2 head-on cut are documented sensitivity variants, never silently
# substituted.
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
