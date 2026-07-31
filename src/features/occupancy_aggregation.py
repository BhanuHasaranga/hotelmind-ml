"""Derives a daily occupancy/revenue "mart" directly from the warehouse
parquet fact/dim tables in pandas.

No pre-aggregated mart (mart_occupancy_daily / mart_revenue_daily) exists in
the warehouse parquet set — only per-booking grain (fact_booking). This
module expands each non-cancelled booking's check-in -> check-out range into
one row per occupied night, then aggregates by (hotel, date).

ASSUMPTION: fact_booking has no room-inventory count, so total_rooms per
hotel is a fixed, documented capacity constant (not derived from data). See
reports/final_phase4/known_limitations.md.
"""

import numpy as np
import pandas as pd

from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Documented assumption: no room-inventory data exists anywhere in the
# source dataset or warehouse. These are illustrative fixed capacities per
# hotel, not measured values.
ASSUMED_TOTAL_ROOMS: dict[str, int] = {"Resort Hotel": 200, "City Hotel": 300}

# Bookings with these statuses never occupied a room and are excluded from
# occupancy/revenue aggregation. NOTE: fact_booking.is_terminal is True for
# every row (it also covers completed Check-Out stays), so it cannot be used
# as an occupancy filter -- reservation_status is the correct field.
NON_OCCUPYING_STATUSES = {"Canceled", "No-Show"}


def expand_booking_nights(fact_booking: pd.DataFrame, dim_date: pd.DataFrame) -> pd.DataFrame:
    """One row per (branch_key, date_key) night a booking occupies.

    A booking with check_in_date_key=D1 and check_out_date_key=D2 occupies
    nights [D1, D2) -- D2 itself is a checkout day, not an occupied night.
    Bookings with nights == 0 contribute no rows. Cancelled/no-show bookings
    are excluded entirely (they never occupied a room).
    """
    occupying = fact_booking[
        ~fact_booking["reservation_status"].isin(NON_OCCUPYING_STATUSES)
        & (fact_booking["nights"] > 0)
    ].copy()

    date_lookup = dim_date.set_index("date_key")["full_date"]

    rows = []
    for branch_key, check_in_key, nights, adr in zip(
        occupying["branch_key"], occupying["check_in_date_key"],
        occupying["nights"], occupying["avg_daily_rate"],
    ):
        if check_in_key not in date_lookup.index:
            continue
        start = date_lookup.loc[check_in_key]
        occupied_dates = pd.date_range(start=start, periods=int(nights), freq="D")
        rows.append(pd.DataFrame({
            "branch_key": branch_key, "full_date": occupied_dates, "avg_daily_rate": adr,
        }))

    if not rows:
        return pd.DataFrame(columns=["branch_key", "full_date", "avg_daily_rate"])

    expanded = pd.concat(rows, ignore_index=True)
    logger.info("expand_booking_nights: %d occupied-room-nights from %d bookings", len(expanded), len(occupying))
    return expanded


def build_daily_occupancy(
    fact_booking: pd.DataFrame, dim_date: pd.DataFrame, dim_hotel: pd.DataFrame
) -> pd.DataFrame:
    """Daily (hotel, date) occupancy/revenue aggregate, the new "mart"
    equivalent computed in-process from warehouse parquet, no Postgres.

    Columns: branch_id(=hotel_key), branch_name, occupancy_date, occupied_rooms,
    total_rooms, occupancy_pct, total_revenue, revenue_7day_avg, occupancy_7day_avg,
    occupancy_30day_avg.
    """
    expanded = expand_booking_nights(fact_booking, dim_date)
    if expanded.empty:
        return pd.DataFrame(columns=[
            "branch_id", "branch_name", "occupancy_date", "occupied_rooms", "total_rooms",
            "occupancy_pct", "total_revenue", "revenue_7day_avg", "occupancy_7day_avg", "occupancy_30day_avg",
        ])

    daily = expanded.groupby(["branch_key", "full_date"]).agg(
        occupied_rooms=("avg_daily_rate", "size"),
        total_revenue=("avg_daily_rate", "sum"),
        avg_daily_rate=("avg_daily_rate", "mean"),
    ).reset_index()

    daily = daily.merge(dim_hotel[["hotel_key", "hotel_name"]], left_on="branch_key", right_on="hotel_key")
    daily["total_rooms"] = daily["hotel_name"].map(ASSUMED_TOTAL_ROOMS)
    daily["occupancy_pct"] = (daily["occupied_rooms"] / daily["total_rooms"] * 100).clip(upper=100.0)

    daily = daily.rename(columns={"branch_key": "branch_id", "hotel_name": "branch_name", "full_date": "occupancy_date"})
    daily = daily.sort_values(["branch_id", "occupancy_date"]).reset_index(drop=True)

    daily["revenue_7day_avg"] = (
        daily.groupby("branch_id")["total_revenue"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        .fillna(daily["total_revenue"])
    )
    daily["occupancy_7day_avg"] = (
        daily.groupby("branch_id")["occupancy_pct"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        .fillna(daily["occupancy_pct"])
    )
    daily["occupancy_30day_avg"] = (
        daily.groupby("branch_id")["occupancy_pct"]
        .transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
        .fillna(daily["occupancy_pct"])
    )

    daily = daily.drop(columns=["hotel_key"])
    logger.info("build_daily_occupancy: %d (branch, date) rows", len(daily))
    return daily


def load_daily_occupancy(branch_id: int | None = None) -> pd.DataFrame:
    """Convenience loader: reads the warehouse parquet tables and builds the
    daily occupancy/revenue aggregate, optionally filtered to one branch.
    Shared by Occupancy, Pricing, Restaurant, and Staffing pipelines so each
    doesn't repeat the parquet-read + join boilerplate.
    """
    fact_booking = pd.read_parquet(settings.data_warehouse_dir_path / "fact_booking.parquet")
    dim_date = pd.read_parquet(settings.data_warehouse_dir_path / "dim_date.parquet")
    dim_hotel = pd.read_parquet(settings.data_warehouse_dir_path / "dim_hotel.parquet")

    daily = build_daily_occupancy(fact_booking, dim_date, dim_hotel)
    if branch_id is not None:
        daily = daily[daily["branch_id"] == branch_id].reset_index(drop=True)
    return daily
