"""Synthetic daily seed datasets for Restaurant and Staffing.

No real restaurant-order or staff-attendance data exists anywhere in this
project (confirmed: not in hotel_bookings.csv, not in the warehouse, no
mart_restaurant_daily/mart_staff_daily was ever populated). These generators
produce small, clearly-labeled synthetic seeds -- driven by real derived
daily occupancy (see src/features/occupancy_aggregation.py), not pure noise
-- so Restaurant/Staffing models can be trained on deterministic, reproducible
data. Prominently documented as fabricated in
reports/final_phase4/known_limitations.md; NOT to be treated as real
business data.

Column shapes are designed to match what the existing (unmodified)
src/features/restaurant_features.py::derive_meal_quantities and
src/pipelines/staffing_pipeline.py already expect.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.constants import MEAL_PERIODS, STAFF_DEPARTMENTS
from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

RESTAURANT_SEED_FILENAME = "restaurant_daily_synthetic.csv"
STAFFING_SEED_FILENAME = "staffing_daily_synthetic.csv"

# Documented, fixed synthetic-generation constants (not derived from data).
MEAL_BASE_REVENUE_PER_ROOM = {"breakfast": 8.0, "lunch": 6.0, "dinner": 12.0}
MEAL_AVG_ITEM_VALUE = {"breakfast": 9.5, "lunch": 14.0, "dinner": 22.0}

DEPARTMENT_STAFF_RATIO = {"Reception": 0.03, "Kitchen": 0.05, "Housekeeping": 0.08}
DEPARTMENT_ID_MAP = {name: idx + 1 for idx, name in enumerate(STAFF_DEPARTMENTS)}

_WEEKEND_MULTIPLIER = 1.15
_WEEKDAY_MULTIPLIER = 1.0


def _day_of_week_multiplier(dates: pd.Series) -> pd.Series:
    is_weekend = pd.to_datetime(dates).dt.dayofweek.isin([5, 6])
    return is_weekend.map({True: _WEEKEND_MULTIPLIER, False: _WEEKDAY_MULTIPLIER})


def generate_restaurant_daily(daily_occupancy: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """One row per (branch_id, date): total_revenue, {meal}_revenue for each
    meal, items_sold, avg_item_value. Deterministic given `seed` and the
    input occupancy DataFrame -- driven by occupied_rooms, not pure noise.
    """
    rng = np.random.default_rng(seed)
    out = daily_occupancy[["branch_id", "occupancy_date", "occupied_rooms"]].copy()
    out = out.rename(columns={"occupancy_date": "date"})
    multiplier = _day_of_week_multiplier(out["date"])

    total_revenue = pd.Series(0.0, index=out.index)
    for meal in MEAL_PERIODS:
        noise = 1.0 + rng.normal(0, 0.05, size=len(out))
        revenue = MEAL_BASE_REVENUE_PER_ROOM[meal] * out["occupied_rooms"] * multiplier * noise
        revenue = revenue.clip(lower=0.0)
        out[f"{meal}_revenue"] = revenue.round(2)
        total_revenue += revenue

    out["total_revenue"] = total_revenue.round(2)
    out["avg_item_value"] = sum(MEAL_AVG_ITEM_VALUE.values()) / len(MEAL_AVG_ITEM_VALUE)
    out["items_sold"] = (out["total_revenue"] / out["avg_item_value"]).round().astype(int)
    out["total_orders"] = out["items_sold"]

    out = out.drop(columns=["occupied_rooms"])
    logger.info("generate_restaurant_daily: %d rows (seed=%d)", len(out), seed)
    return out


def generate_staffing_daily(daily_occupancy: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """One row per (branch_id, date, department): scheduled_employees,
    present_employees. Deterministic given `seed`, driven by occupied_rooms.
    """
    rng = np.random.default_rng(seed)
    base = daily_occupancy[["branch_id", "occupancy_date", "occupied_rooms"]].copy()
    base = base.rename(columns={"occupancy_date": "date"})
    multiplier = _day_of_week_multiplier(base["date"])

    rows = []
    for department in STAFF_DEPARTMENTS:
        dept_df = base.copy()
        dept_df["department_name"] = department
        dept_df["department_id"] = DEPARTMENT_ID_MAP[department]

        expected = DEPARTMENT_STAFF_RATIO[department] * dept_df["occupied_rooms"] * multiplier
        noise = rng.integers(-1, 2, size=len(dept_df))  # -1, 0, or 1
        present = np.ceil(expected).to_numpy().astype(int) + noise
        dept_df["present_employees"] = np.clip(present, 0, None)
        dept_df["scheduled_employees"] = np.clip(np.ceil(expected).to_numpy().astype(int), 0, None)

        rows.append(dept_df.drop(columns=["occupied_rooms"]))

    out = pd.concat(rows, ignore_index=True)
    logger.info("generate_staffing_daily: %d rows across %d departments (seed=%d)", len(out), len(STAFF_DEPARTMENTS), seed)
    return out


def ensure_restaurant_seed(daily_occupancy: pd.DataFrame) -> Path:
    path = settings.data_raw_dir_path / RESTAURANT_SEED_FILENAME
    if not path.exists():
        df = generate_restaurant_daily(daily_occupancy)
        df.to_csv(path, index=False)
        logger.info("Generated synthetic seed: %s", path)
    return path


def ensure_staffing_seed(daily_occupancy: pd.DataFrame) -> Path:
    path = settings.data_raw_dir_path / STAFFING_SEED_FILENAME
    if not path.exists():
        df = generate_staffing_daily(daily_occupancy)
        df.to_csv(path, index=False)
        logger.info("Generated synthetic seed: %s", path)
    return path
