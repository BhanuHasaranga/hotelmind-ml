"""Warehouse loading pipeline: transforms the cleaned dataset into the
Phase 3 star-schema shape (dim_date, dim_hotel, dim_room_type, dim_guest,
fact_booking) and writes it locally under data/warehouse/. Optionally loads
into the real Postgres warehouse via --write-db, failing gracefully if the
target schema isn't provisioned.

fact_restaurant_sale and fact_staff_attendance are out of scope: no data in
the canonical dataset supports them, and they are intentionally untouched.

CLI: python -m src.pipelines.warehouse_loader [--write-db]
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.config.settings import settings
from src.database.postgres import get_connection
from src.database.writer import WarehouseWriteError, bulk_upsert
from src.pipelines import keygen
from src.pipelines.warehouse_reports import (
    write_key_generation,
    write_lineage_report,
    write_loading_summary,
    write_mapping_summary,
    write_validation_report,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

CLEANED_PARQUET = "hotel_bookings_clean.parquet"

FACT_SORT_KEYS = ["arrival_date", "hotel", "lead_time", "adr"]

# Columns from the cleaned dataset used to build a per-row reservation_id
# hash (see keygen.reservation_id).
RESERVATION_HASH_COLUMNS = [
    "hotel", "is_canceled", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_week_number", "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies", "meal", "country",
    "market_segment", "distribution_channel", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "reserved_room_type", "assigned_room_type",
    "booking_changes", "deposit_type", "agent", "company", "days_in_waiting_list",
    "customer_type", "adr", "required_car_parking_spaces", "total_of_special_requests",
    "reservation_status", "reservation_status_date",
]


def _load_room_type_seed() -> pd.DataFrame:
    seed_path = settings.data_raw_dir_path / "room_type_dim.csv"
    return pd.read_csv(seed_path)


def _nights(df: pd.DataFrame) -> pd.Series:
    return df["stays_in_weekend_nights"].fillna(0).astype(int) + df["stays_in_week_nights"].fillna(0).astype(int)


def build_dim_date(df: pd.DataFrame) -> pd.DataFrame:
    checkin = df["arrival_date"]
    checkout = df["arrival_date"] + pd.to_timedelta(_nights(df), unit="D")
    all_dates = pd.concat([checkin, checkout]).dropna().dt.normalize().unique()

    dates = pd.to_datetime(pd.Series(all_dates)).sort_values().reset_index(drop=True)
    out = pd.DataFrame({
        "date_key": dates.map(keygen.date_id),
        "full_date": dates,
        "year": dates.dt.year,
        "month": dates.dt.month,
        "month_name": dates.dt.month_name(),
        "day": dates.dt.day,
        "day_of_week": dates.dt.dayofweek,
        "week_of_year": dates.dt.isocalendar().week.astype(int),
        "quarter": dates.dt.quarter,
        "is_weekend": dates.dt.dayofweek.isin([5, 6]),
    })
    out = out.drop_duplicates(subset=["date_key"]).reset_index(drop=True)
    logger.info("build_dim_date: %d distinct dates", len(out))
    return out


def build_dim_hotel(df: pd.DataFrame) -> pd.DataFrame:
    hotels = sorted(df["hotel"].dropna().unique().tolist())
    out = pd.DataFrame({
        "hotel_key": [keygen.hotel_id(h) for h in hotels],
        "hotel_id": ["RESORT" if h == "Resort Hotel" else "CITY" for h in hotels],
        "hotel_name": hotels,
    })
    logger.info("build_dim_hotel: %d hotels", len(out))
    return out


def build_dim_room_type(df: pd.DataFrame) -> pd.DataFrame:
    seed = _load_room_type_seed()
    codes = df["assigned_room_type"].dropna().astype(str).str.strip().str.upper()
    mapped_ids = codes.map(keygen.room_type_id)

    examples: dict[int, set[str]] = {}
    for code, room_id in zip(codes, mapped_ids):
        examples.setdefault(room_id, set()).add(code)

    rows = []
    for _, seed_row in seed.iterrows():
        rid = int(seed_row["room_type_id"])
        rows.append({
            "room_type_key": rid,
            "room_type_id": rid,
            "room_type_name": seed_row["room_type_name"],
            "base_price_multiplier": seed_row["base_price_multiplier"],
            "source_code_examples": ", ".join(sorted(examples.get(rid, []))),
        })

    if keygen.UNMAPPED_ROOM_TYPE_ID in examples:
        rows.append({
            "room_type_key": keygen.UNMAPPED_ROOM_TYPE_ID,
            "room_type_id": keygen.UNMAPPED_ROOM_TYPE_ID,
            "room_type_name": "Unmapped",
            "base_price_multiplier": None,
            "source_code_examples": ", ".join(sorted(examples[keygen.UNMAPPED_ROOM_TYPE_ID])),
        })

    out = pd.DataFrame(rows)
    logger.info("build_dim_room_type: %d room types", len(out))
    return out


def build_dim_guest(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["guest_key"] = working.apply(keygen.guest_id, axis=1)
    working["_nights"] = _nights(working)
    working["_spend"] = working["_nights"] * working["adr"]

    grouped = working.groupby("guest_key").agg(
        nationality=("country", "first"),
        lifetime_bookings=("guest_key", "size"),
        lifetime_spend=("_spend", "sum"),
        first_stay_date=("arrival_date", "min"),
        last_stay_date=("arrival_date", "max"),
    ).reset_index()

    grouped["guest_id"] = grouped["guest_key"]
    grouped["full_name"] = pd.NA

    out = grouped[[
        "guest_key", "guest_id", "full_name", "nationality",
        "lifetime_bookings", "lifetime_spend", "first_stay_date", "last_stay_date",
    ]]
    logger.info("build_dim_guest: %d distinct guest clusters from %d rows", len(out), len(df))
    return out


def build_fact_booking(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    working = df.sort_values(FACT_SORT_KEYS, kind="mergesort").reset_index(drop=True)
    working["_row_index"] = working.index

    nights = _nights(working)
    checkout_date = working["arrival_date"] + pd.to_timedelta(nights, unit="D")

    fact = pd.DataFrame({
        "surrogate_key": working.apply(
            lambda r: keygen.reservation_id(r, r["_row_index"], RESERVATION_HASH_COLUMNS), axis=1
        ),
        "reservation_status": working["reservation_status"],
        "room_key": working["assigned_room_type"].astype(str).str.strip().str.upper().map(keygen.room_type_id),
        "branch_key": working["hotel"].map(keygen.branch_id),
        "guest_key": working.apply(keygen.guest_id, axis=1),
        "check_in_date_key": working["arrival_date"].map(keygen.date_id),
        "check_out_date_key": checkout_date.map(keygen.date_id),
        "nights": nights,
        "adults": working["adults"],
        "children": working["children"],
        "avg_daily_rate": working["adr"],
        "is_adults_outlier": working["is_adults_outlier"],
    })
    fact["reservation_id"] = fact["surrogate_key"]
    fact["total_amount"] = fact["nights"] * fact["avg_daily_rate"]
    fact["is_terminal"] = fact["reservation_status"].isin(["Canceled", "Check-Out", "No-Show"])
    fact["is_completed"] = fact["reservation_status"] == "Check-Out"

    fact = fact[[
        "surrogate_key", "reservation_id", "reservation_status", "room_key", "branch_key",
        "guest_key", "check_in_date_key", "check_out_date_key", "nights", "adults",
        "children", "total_amount", "avg_daily_rate", "is_terminal", "is_completed",
        "is_adults_outlier",
    ]]

    rows_before = len(fact)
    unresolvable = fact["check_in_date_key"].isna() | fact["check_out_date_key"].isna()
    excluded = int(unresolvable.sum())
    fact = fact[~unresolvable].reset_index(drop=True)

    stats = {
        "rows_before": rows_before,
        "rows_after": len(fact),
        "excluded_unresolvable_date_fk": excluded,
    }
    if excluded:
        logger.warning("build_fact_booking: excluded %d rows with unresolvable date FK", excluded)
    logger.info("build_fact_booking: %d rows", len(fact))
    return fact, stats


def build_all(clean_df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    dim_date = build_dim_date(clean_df)
    dim_hotel = build_dim_hotel(clean_df)
    dim_room_type = build_dim_room_type(clean_df)
    dim_guest = build_dim_guest(clean_df)
    fact_booking, fact_stats = build_fact_booking(clean_df)

    tables = {
        "dim_date": dim_date,
        "dim_hotel": dim_hotel,
        "dim_room_type": dim_room_type,
        "dim_guest": dim_guest,
        "fact_booking": fact_booking,
    }
    return tables, fact_stats


def save_local(tables: dict[str, pd.DataFrame]) -> dict[str, Path]:
    paths = {}
    for name, table in tables.items():
        parquet_path = settings.data_warehouse_dir_path / f"{name}.parquet"
        csv_path = settings.data_warehouse_dir_path / f"{name}.csv"
        table.to_parquet(parquet_path, index=False)
        table.to_csv(csv_path, index=False)
        paths[name] = parquet_path
    logger.info("save_local: wrote %d warehouse tables to %s", len(tables), settings.data_warehouse_dir_path)
    return paths


PK_COLUMNS = {
    "dim_date": ["date_key"],
    "dim_hotel": ["hotel_key"],
    "dim_room_type": ["room_type_key"],
    "dim_guest": ["guest_key"],
    "fact_booking": ["surrogate_key"],
}

LOAD_ORDER = ["dim_date", "dim_hotel", "dim_room_type", "dim_guest", "fact_booking"]


def load_to_db(tables: dict[str, pd.DataFrame]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    try:
        conn = get_connection()
    except Exception as exc:
        error_summary = " ".join(str(exc).split())
        logger.error(
            "Postgres unreachable (%s) — local warehouse files were still written successfully to %s",
            error_summary, settings.data_warehouse_dir_path,
        )
        for name in LOAD_ORDER:
            results[name] = {"rows_written": 0, "note": f"connection failed: {error_summary}"}
        return results

    try:
        for name in LOAD_ORDER:
            table = tables[name]
            try:
                rows_written = bulk_upsert(conn, name, table, PK_COLUMNS[name])
                conn.commit()
                results[name] = {"rows_written": rows_written, "note": "loaded"}
            except WarehouseWriteError as exc:
                conn.rollback()
                logger.error(str(exc))
                results[name] = {"rows_written": 0, "note": "table not found"}
    finally:
        conn.close()

    return results


def validate_warehouse(tables: dict[str, pd.DataFrame]) -> dict:
    results: dict = {"tables": {}, "fk_checks": {}}

    required_not_null = {
        "fact_booking": ["surrogate_key", "room_key", "branch_key", "guest_key", "check_in_date_key"],
    }

    for name, table in tables.items():
        pk_cols = PK_COLUMNS[name]
        pk_series = table[pk_cols[0]] if len(pk_cols) == 1 else table[pk_cols].apply(tuple, axis=1)
        null_violations = {}
        for col in required_not_null.get(name, []):
            null_violations[col] = int(table[col].isna().sum())

        results["tables"][name] = {
            "row_count": len(table),
            "pk_unique": bool(pk_series.is_unique),
            "null_violations": null_violations,
        }

    fact = tables["fact_booking"]
    fk_map = {
        "fact_booking.room_key -> dim_room_type.room_type_key": (fact["room_key"], tables["dim_room_type"]["room_type_key"]),
        "fact_booking.branch_key -> dim_hotel.hotel_key": (fact["branch_key"], tables["dim_hotel"]["hotel_key"]),
        "fact_booking.guest_key -> dim_guest.guest_key": (fact["guest_key"], tables["dim_guest"]["guest_key"]),
        "fact_booking.check_in_date_key -> dim_date.date_key": (fact["check_in_date_key"], tables["dim_date"]["date_key"]),
        "fact_booking.check_out_date_key -> dim_date.date_key": (fact["check_out_date_key"], tables["dim_date"]["date_key"]),
    }

    for fk_name, (child, parent) in fk_map.items():
        orphans = set(child.dropna().unique()) - set(parent.dropna().unique())
        results["fk_checks"][fk_name] = {
            "orphan_count": len(orphans),
            "sample_orphans": list(orphans)[:5],
        }

    return results


MAPPING_ASSUMPTIONS = {
    "Hotel / branch identity": (
        "hotel_key and branch_key both come from a fixed lookup "
        "(`Resort Hotel` -> 1, `City Hotel` -> 2). The canonical dataset has no "
        "branch granularity below hotel, so branch_key reuses hotel_key."
    ),
    "Room type mapping": (
        "The raw dataset's reserved/assigned_room_type columns are single-letter "
        "codes (A, B, C, ...) with no authoritative mapping to the seed "
        "room_type_dim.csv names (Standard/Deluxe/Suite). An ordinal-tier "
        "assumption is used (A-C -> Standard, D-G -> Deluxe, H/L -> Suite); any "
        "other code falls into an explicit 'Unmapped' bucket (room_type_id=0) "
        "rather than being dropped. This mapping should be reviewed by a "
        "domain stakeholder before being treated as ground truth."
    ),
    "Guest identity": (
        "The anonymized source data has no true guest identity. guest_key is a "
        "hash of (country, market_segment, distribution_channel, customer_type, "
        "is_repeated_guest, agent, company) — this identifies a 'guest profile "
        "cluster', not a real unique person. Many distinct real guests will "
        "collide into the same guest_key; this is an accepted limitation, not a "
        "bug. Party composition (adults/children/babies) was deliberately "
        "excluded from the hash since it varies trip-to-trip for the same guest "
        "and would fragment rather than cluster."
    ),
    "Reservation identity": (
        "reservation_id/surrogate_key is a hash of all cleaned-row values plus a "
        "stable post-sort row index (fact rows are sorted by "
        "arrival_date/hotel/lead_time/adr with a stable mergesort before key "
        "assignment), guarding against two bookings that share identical values "
        "across every column."
    ),
    "total_amount derivation": (
        "total_amount = nights * adr. No paid_amount/outstanding_amount data "
        "exists anywhere in the canonical dataset; those fields are simply not "
        "populated by this loader rather than fabricated as 0 or a misleading NULL."
    ),
    "Out-of-scope tables": (
        "fact_restaurant_sale and fact_staff_attendance are not populated — no "
        "data in any of the source CSVs supports them. They remain untouched, "
        "as instructed, and are not created as empty/stub files."
    ),
}

KEYGEN_NOTES = {
    "date_key": "int(YYYYMMDD) from the calendar date.",
    "hotel_key / branch_key": "Fixed dict lookup: {'Resort Hotel': 1, 'City Hotel': 2}.",
    "room_type_key": "Letter code -> tier id via ROOM_TYPE_CODE_MAP, else 0 ('Unmapped').",
    "guest_key": "SHA-256 hash of (country, market_segment, distribution_channel, customer_type, is_repeated_guest, agent, company), truncated to 63 bits.",
    "surrogate_key / reservation_id": "SHA-256 hash of all cleaned-row values plus a stable post-sort row index.",
}


def run(write_db: bool = False) -> dict:
    start = time.perf_counter()
    exec_times = {}

    clean_path = settings.data_processed_dir_path / CLEANED_PARQUET
    if not clean_path.exists():
        raise FileNotFoundError(
            f"{clean_path} not found. Run `python -m src.pipelines.data_cleaning` first."
        )

    t0 = time.perf_counter()
    clean_df = pd.read_parquet(clean_path)
    exec_times["load_cleaned"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tables, fact_stats = build_all(clean_df)
    exec_times["build_all"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    save_local(tables)
    exec_times["save_local"] = time.perf_counter() - t0

    db_results = None
    if write_db:
        t0 = time.perf_counter()
        db_results = load_to_db(tables)
        exec_times["load_to_db"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    validation_results = validate_warehouse(tables)
    exec_times["validate"] = time.perf_counter() - t0

    stats_path = settings.warehouse_loading_reports_dir_path / "_cleaning_stats.json"
    cleaning_stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {
        "loaded_rows": None, "cleaned_rows": len(clean_df), "warnings": [], "errors": [], "steps": {},
    }

    table_row_counts = {name: len(t) for name, t in tables.items()}

    working = clean_df.copy()
    guest_keys = working.apply(keygen.guest_id, axis=1)
    codes = working["assigned_room_type"].dropna().astype(str).str.strip().str.upper()
    unmapped_codes = set(codes[codes.map(keygen.room_type_id) == keygen.UNMAPPED_ROOM_TYPE_ID])
    guest_cluster_stats = {
        "distinct_guest_keys": int(guest_keys.nunique()),
        "total_rows": len(working),
        "avg_rows_per_cluster": len(working) / max(guest_keys.nunique(), 1),
    }

    write_loading_summary(cleaning_stats, table_row_counts, exec_times)
    write_key_generation(KEYGEN_NOTES, unmapped_codes, guest_cluster_stats)
    write_validation_report(validation_results)
    write_mapping_summary(MAPPING_ASSUMPTIONS)
    write_lineage_report(
        raw_count=cleaning_stats.get("loaded_rows"),
        cleaned_count=len(clean_df),
        warehouse_counts=table_row_counts,
        db_counts=db_results,
    )

    elapsed = time.perf_counter() - start
    logger.info(
        "warehouse_loader complete: rows=%s elapsed=%.2fs write_db=%s",
        table_row_counts, elapsed, write_db,
    )

    return {
        "table_row_counts": table_row_counts,
        "fact_stats": fact_stats,
        "validation_results": validation_results,
        "db_results": db_results,
        "execution_seconds": elapsed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-db", action="store_true")
    args = parser.parse_args()
    run(write_db=args.write_db)
