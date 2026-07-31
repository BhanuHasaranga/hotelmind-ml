"""Deterministic surrogate key generation for the warehouse loader.

Every function here is a pure function of its inputs: the same input always
produces the same key, with no randomness, no UUIDs, and no run-order
dependence (beyond the caller supplying a stable row_index — see
reservation_id). This is what lets warehouse_loader.py be re-run safely
without keys drifting between runs.
"""

import hashlib

import pandas as pd

# Only two hotel values are present in the canonical dataset ("Resort Hotel",
# "City Hotel"). A fixed lookup gives small, memorable IDs; an unknown value
# raises, since it would signal the source data shape changed in a way this
# pipeline hasn't been reviewed for.
HOTEL_ID_MAP: dict[str, int] = {"Resort Hotel": 1, "City Hotel": 2}

# The raw dataset's reserved/assigned_room_type columns are single-letter
# codes (A, B, C, ...) that do not natively align with the seed
# room_type_dim.csv names (Standard/Deluxe/Suite). This ordinal-tier mapping
# is a documented assumption, not derived from any authoritative source, and
# should be reviewed by a domain stakeholder before being treated as ground
# truth (see reports/warehouse_loading/mapping_summary.md).
ROOM_TYPE_CODE_MAP: dict[str, int] = {
    "A": 1, "B": 1, "C": 1,           # -> Standard
    "D": 2, "E": 2, "F": 2, "G": 2,   # -> Deluxe
    "H": 3, "L": 3,                   # -> Suite
}
UNMAPPED_ROOM_TYPE_ID = 0  # explicit "Unmapped" bucket for any other code


def stable_hash_int(*parts: object, bits: int = 63) -> int:
    """Deterministic positive integer from the given parts.

    SHA-256 of the pipe-joined string representation of parts, truncated to
    `bits` bits so the result always fits a signed BIGINT. No salt, no
    randomness: identical parts always yield the identical integer, across
    processes and runs.
    """
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return value & ((1 << bits) - 1)


def date_id(dt: pd.Timestamp) -> int | None:
    """YYYYMMDD integer key for a date. Returns None for NaT (caller decides
    how to handle unresolvable date FKs)."""
    if pd.isna(dt):
        return None
    return int(pd.Timestamp(dt).strftime("%Y%m%d"))


def hotel_id(hotel_name: str) -> int:
    """Fixed lookup. Raises KeyError for any hotel name outside the known,
    closed set of 2 values — treated as a real data-shape regression."""
    return HOTEL_ID_MAP[hotel_name]


def branch_id(hotel_name: str) -> int:
    """Reuses hotel_id: the source data has no branch granularity below
    hotel (Resort Hotel / City Hotel are themselves the only 'branches')."""
    return hotel_id(hotel_name)


def room_type_id(code: str) -> int:
    """Maps a raw letter room-type code to the seed dim's tier id. Unmapped
    codes fall back to UNMAPPED_ROOM_TYPE_ID rather than raising, since
    unseen codes are expected/documented, not an error condition."""
    return ROOM_TYPE_CODE_MAP.get(str(code).strip().upper(), UNMAPPED_ROOM_TYPE_ID)


# Fields used to derive a guest identity proxy. The anonymized source data
# carries no true guest identity, so this hash clusters rows that share a
# booking "profile" (same country, channel, agent/company, repeat-guest
# flag) rather than identifying a real unique person — many distinct real
# guests will collide into the same guest_key. This is a documented,
# accepted limitation, not a bug (see mapping_summary.md). Party composition
# (adults/children/babies) is deliberately excluded: it varies trip-to-trip
# for the same guest and would fragment rather than cluster.
GUEST_KEY_FIELDS = [
    "country",
    "market_segment",
    "distribution_channel",
    "customer_type",
    "is_repeated_guest",
    "agent",
    "company",
]


def guest_id(row: pd.Series) -> int:
    parts = [row.get(field) for field in GUEST_KEY_FIELDS]
    return stable_hash_int(*parts)


def reservation_id(row: pd.Series, row_index: int, columns: list[str]) -> int:
    """Deterministic per-row key.

    Hashing row content alone is not sufficient: two genuinely different
    bookings can share identical values across every cleaned column. To
    guarantee per-row uniqueness while staying deterministic across re-runs,
    the hash also includes a stable row_index. Callers must assign
    row_index from a DataFrame sorted by a fixed key (see
    warehouse_loader.build_fact_booking) so that "row N in cleaned output"
    is reproducible run-to-run given identical input data.
    """
    parts = [str(row[c]) for c in columns] + [str(row_index)]
    return stable_hash_int(*parts)
