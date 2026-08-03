"""Reads local mart-equivalent snapshots for `DATA_SOURCE=local`.

Mirrors the pattern in `src/prediction/predict_churn.py` (read parquet from
`settings.data_warehouse_dir_path`). If the mart-equivalent CSV/parquet
files don't exist locally yet (the live warehouse only populates the raw
star-schema tables, not the marts, in this repo's local snapshot), small
synthetic mart snapshots are generated on first access, following the
precedent of `src/pipelines/synthetic_data.py`.
"""

import numpy as np
import pandas as pd

from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

MART_FILES = {
    "mart_revenue_daily": "mart_revenue_daily.csv",
    "mart_occupancy_daily": "mart_occupancy_daily.csv",
    "mart_restaurant_daily": "mart_restaurant_daily.csv",
    "mart_staff_daily": "mart_staff_daily.csv",
}


def _synthesize_mart_occupancy_daily(seed: int = 42) -> pd.DataFrame:
    fact_booking = pd.read_parquet(settings.data_warehouse_dir_path / "fact_booking.parquet")
    dim_hotel = pd.read_parquet(settings.data_warehouse_dir_path / "dim_hotel.parquet")
    rng = np.random.default_rng(seed)

    branch_ids = sorted(dim_hotel["hotel_key"].unique())[:5] or [1]
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=90, freq="D")

    rows = []
    for branch_id in branch_ids:
        base = rng.uniform(55, 85)
        for i, date in enumerate(dates):
            noise = rng.normal(0, 4)
            occ_pct = float(np.clip(base + 5 * np.sin(i / 7) + noise, 20, 99))
            total_rooms = 100
            occupied = int(round(total_rooms * occ_pct / 100))
            rows.append(
                {
                    "branch_id": branch_id,
                    "date": date.date().isoformat(),
                    "occupancy_pct": round(occ_pct, 2),
                    "occupancy_7day_avg": round(occ_pct, 2),
                    "occupancy_30day_avg": round(occ_pct, 2),
                    "total_rooms": total_rooms,
                    "occupied_rooms": occupied,
                }
            )
    return pd.DataFrame(rows)


def _synthesize_mart_revenue_daily(occupancy_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    df = occupancy_df.copy()
    adr = rng.uniform(80, 160, size=len(df))
    df["avg_daily_rate"] = adr.round(2)
    df["room_revenue"] = (df["occupied_rooms"] * df["avg_daily_rate"]).round(2)
    df["fb_revenue"] = (df["room_revenue"] * rng.uniform(0.15, 0.3, size=len(df))).round(2)
    df["total_revenue"] = (df["room_revenue"] + df["fb_revenue"]).round(2)
    df["revpar"] = (df["total_revenue"] / df["total_rooms"]).round(2)
    df["revenue_7day_avg"] = df["total_revenue"].round(2)
    return df[
        [
            "branch_id",
            "date",
            "total_revenue",
            "room_revenue",
            "fb_revenue",
            "avg_daily_rate",
            "revpar",
            "occupancy_pct",
            "revenue_7day_avg",
        ]
    ]


def _synthesize_mart_restaurant_daily(occupancy_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    df = occupancy_df[["branch_id", "date", "occupied_rooms"]].copy()
    df["total_orders"] = (df["occupied_rooms"] * rng.uniform(1.5, 2.5, size=len(df))).round().astype(int)
    df["items_sold"] = (df["total_orders"] * rng.uniform(1.2, 1.8, size=len(df))).round().astype(int)
    df["avg_order_value"] = rng.uniform(15, 35, size=len(df)).round(2)
    df["total_revenue"] = (df["total_orders"] * df["avg_order_value"]).round(2)
    df["breakfast_revenue"] = (df["total_revenue"] * 0.3).round(2)
    df["lunch_revenue"] = (df["total_revenue"] * 0.35).round(2)
    df["dinner_revenue"] = (df["total_revenue"] * 0.35).round(2)
    df["revenue_7day_avg"] = df["total_revenue"].round(2)
    return df.drop(columns=["occupied_rooms"])


def _synthesize_mart_staff_daily(occupancy_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 3)
    departments = ["Reception", "Kitchen", "Housekeeping"]
    rows = []
    for _, row in occupancy_df.iterrows():
        for dept_id, dept in enumerate(departments, start=1):
            scheduled = int(max(1, round(row["occupied_rooms"] * 0.04 * rng.uniform(0.8, 1.2))))
            present = max(0, scheduled - rng.integers(0, 2))
            actual_hours = present * 8 * rng.uniform(0.9, 1.05)
            rows.append(
                {
                    "branch_id": row["branch_id"],
                    "department_id": dept_id,
                    "department_name": dept,
                    "date": row["date"],
                    "scheduled_employees": scheduled,
                    "present_employees": present,
                    "attendance_rate_pct": round(100 * present / scheduled, 2) if scheduled else 0.0,
                    "total_actual_hours": round(actual_hours, 2),
                    "hours_utilisation_pct": round(100 * actual_hours / (scheduled * 8), 2) if scheduled else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _ensure_mart_snapshots() -> None:
    """Generate local synthetic mart snapshots the first time they're needed."""
    warehouse_dir = settings.data_warehouse_dir_path
    occupancy_path = warehouse_dir / MART_FILES["mart_occupancy_daily"]
    if occupancy_path.exists():
        return

    logger.info("Local mart snapshots not found; generating synthetic snapshots under %s", warehouse_dir)
    occupancy_df = _synthesize_mart_occupancy_daily()
    occupancy_df.to_csv(occupancy_path, index=False)

    revenue_df = _synthesize_mart_revenue_daily(occupancy_df)
    revenue_df.to_csv(warehouse_dir / MART_FILES["mart_revenue_daily"], index=False)

    restaurant_df = _synthesize_mart_restaurant_daily(occupancy_df)
    restaurant_df.to_csv(warehouse_dir / MART_FILES["mart_restaurant_daily"], index=False)

    staff_df = _synthesize_mart_staff_daily(occupancy_df)
    staff_df.to_csv(warehouse_dir / MART_FILES["mart_staff_daily"], index=False)

    logger.info("Synthetic mart snapshots generated (occupancy/revenue/restaurant/staff)")


def read_mart(mart_name: str) -> pd.DataFrame:
    if mart_name not in MART_FILES:
        raise ValueError(f"Unknown mart: {mart_name}; expected one of {sorted(MART_FILES)}")
    _ensure_mart_snapshots()
    path = settings.data_warehouse_dir_path / MART_FILES[mart_name]
    return pd.read_csv(path)
